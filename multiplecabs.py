import os
import math
import random
import time
from datetime import datetime, timedelta

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle

# ------------------ CONFIG ------------------
AVG_SPEED_MPS = 10.0  # ~36 km/h
BBOX = (12.931, 12.936, 77.620, 77.625)  # south, north, west, east (small Bangalore area)
TRAFFIC_BLINKS = 4
TRAFFIC_BLINK_INTERVAL = 0.25
SHORTCUT_MAX_DEG_DIST = 0.0007  # ~ 70–80 meters in lat/lon near Bengaluru
# --------------------------------------------


# ---------- Utilities ----------

def _try_graph_from_bbox(bbox):
    """Load OSM drive graph for bbox, robust to OSMnx v1/v2 signatures."""
    south, north, west, east = bbox
    try:
        # OSMnx v2 preferred signature
        return ox.graph_from_bbox(bbox=bbox, network_type="drive")
    except TypeError:
        # OSMnx v1 signature
        return ox.graph_from_bbox(north, south, east, west, network_type="drive")


def _largest_component(G):
    """Keep the largest weakly connected component to avoid NoPath errors."""
    try:
        return ox.utils_graph.get_largest_component(G, strongly=False)
    except Exception:
        # Fallback: convert to undirected for largest component
        UG = G.to_undirected()
        gc_nodes = max(nx.connected_components(UG), key=len)
        return G.subgraph(gc_nodes).copy()


def _haversine_m(y1, x1, y2, x2):
    """Haversine distance in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(y1), math.radians(y2)
    dphi = math.radians(y2 - y1)
    dlambda = math.radians(x2 - x1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _edge_length_m(G, u, v):
    """Get shortest edge length between u and v in meters, trying both directions if needed."""
    data = G.get_edge_data(u, v)
    if data is None:
        data = G.get_edge_data(v, u)
        if data is None:
            return None
    # pick the minimum-length parallel edge
    best = min(data.values(), key=lambda d: d.get("length", float("inf")))
    return best.get("length", None)


def _compute_path_length_m(G, path):
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        L = _edge_length_m(G, a, b)
        if L is None:
            return None
        total += L
    return total


# ---------- Base map coloring ----------

ARTERIAL_TAGS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}

LOCAL_STREET_TAGS = {"residential", "service", "living_street", "unclassified", "road"}

def classify_edges(G):
    """Return edgelists for (arterial, local_streets, other) and quick-membership sets."""
    arterial, local, other = [], [], []

    def _norm_tag(hw):
        if isinstance(hw, (list, tuple)):
            return str(hw[0])
        return str(hw) if hw is not None else ""

    for u, v, k, d in G.edges(keys=True, data=True):
        tag = _norm_tag(d.get("highway"))
        pair = (u, v)
        if tag in ARTERIAL_TAGS:
            arterial.append(pair)
        elif tag in LOCAL_STREET_TAGS:
            local.append(pair)
        else:
            other.append(pair)

    # sets for quick membership (undirected)
    arterial_set = {tuple(sorted(e)) for e in arterial}
    local_set = {tuple(sorted(e)) for e in local}

    return arterial, local, other, arterial_set, local_set


def draw_base_map(ax, G, pos, arterial, local, other, office_node=None):
    """Draw the full network pre-colored with legend."""
    # background others
    nx.draw_networkx_edges(G, pos, edgelist=other, edge_color="lightgray", width=1.0, ax=ax, alpha=0.6)
    # arterial (orange)
    nx.draw_networkx_edges(G, pos, edgelist=arterial, edge_color="orange", width=1.6, ax=ax, alpha=0.9)
    # local streets (magenta)
    nx.draw_networkx_edges(G, pos, edgelist=local, edge_color="magenta", width=1.3, ax=ax, alpha=0.9)

    # nodes (small, subdued)
    nx.draw_networkx_nodes(G, pos, node_size=6, node_color="#c0c0c0", ax=ax, alpha=0.9)

    # office
    if office_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[office_node], node_color="black", node_size=140, ax=ax)

    # Legend
    legend_elems = [
        Patch(facecolor="orange", edgecolor="orange", label="Main roads (arterials)"),
        Patch(facecolor="magenta", edgecolor="magenta", label="Local streets"),
        Patch(facecolor="lightgray", edgecolor="lightgray", label="Other roads"),
        Patch(facecolor="black", edgecolor="black", label="Office"),
        Patch(facecolor="red", edgecolor="red", label="Traffic incident"),
        Patch(facecolor="none", edgecolor="magenta", label="Temporary shortcut", linewidth=2),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9, frameon=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Pre-colored map: Orange = Arterials, Magenta = Local Streets", fontsize=12)


# ---------- Shortcut injection ----------

def add_temporary_shortcut(G, pos, from_node):
    """
    Add a temporary 'street' (magenta) between the current node and a nearby
    non-neighbor node. Returns list of edges added [(u, w), (w, u)] for drawing.
    """
    x1, y1 = pos[from_node]
    candidates = list(G.nodes())
    random.shuffle(candidates)

    target = None
    for w in candidates[:120]:  # check at most 120 random nodes nearby
        if w == from_node:
            continue
        if G.has_edge(from_node, w) or G.has_edge(w, from_node):
            continue
        x2, y2 = pos[w]
        # ~70–80 meters in degrees near Bengaluru
        if math.hypot(x2 - x1, y2 - y1) <= SHORTCUT_MAX_DEG_DIST:
            target = w
            break

    if target is None:
        return []  # no shortcut found

    # compute an approximate length in meters
    y1lat, x1lon = G.nodes[from_node]["y"], G.nodes[from_node]["x"]
    y2lat, x2lon = G.nodes[target]["y"], G.nodes[target]["x"]
    length_m = _haversine_m(y1lat, x1lon, y2lat, x2lon)

    # Add both directions to keep it usable
    attrs = {
        "length": length_m,
        "highway": "service",     # classify as a local street
        "name": "TemporaryShortcut",
        "shortcut": True,         # mark for styling
    }
    # Add on MultiDiGraph
    G.add_edge(from_node, target, **attrs)
    G.add_edge(target, from_node, **attrs)
    return [(from_node, target), (target, from_node)]


# ---------- Core sim functions ----------

def build_city_graph():
    print(f"Downloading street map for bbox {BBOX}...")
    G = _try_graph_from_bbox(BBOX)
    G = _largest_component(G)
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    return G, pos


def choose_office(G):
    office = random.choice(list(G.nodes))
    print(f"Office is set at node {office}")
    return office


def assign_employees(G, num_agents=3):
    employees = {}
    nodes = list(G.nodes)
    for i in range(num_agents):
        emp = f"Employee_{i+1}"
        employees[emp] = random.choice(nodes)
    return employees


def compute_eta_seconds(G, path):
    length_m = _compute_path_length_m(G, path) or 0.0
    return length_m / AVG_SPEED_MPS


def live_simulation(G, pos, employees, office):
    feedback = {}
    # Pre-classify edges for base-map coloring and quick membership
    arterial, local, other, arterial_set, local_set = classify_edges(G)

    traffic_triggered = False
    shortcut_edges_all = set()  # remember shortcut edges (undirected for coloring)

    for emp, dest in employees.items():
        print(f"\nStarting trip for {emp} (Office -> {dest} -> Office)...")

        # Plan base route (to dest + back)
        try:
            path_fwd = nx.shortest_path(G, office, dest, weight="length")
            path_back = nx.shortest_path(G, dest, office, weight="length")
        except nx.NetworkXNoPath:
            print(f"Could not plan path for {emp} (disconnected). Skipping.")
            continue

        full_path = path_fwd + path_back[1:]
        # ETA at departure
        start_time = datetime.now()
        eta_departure = start_time + timedelta(seconds=compute_eta_seconds(G, full_path))
        print(f"Start time: {start_time.strftime('%H:%M:%S')}")
        print(f"ETA (at departure): {eta_departure.strftime('%H:%M:%S')}")

        # Prepare figure
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_base_map(ax, G, pos, arterial, local, other, office_node=office)

        # Simulated time accumulator (for ATA independent of visualization pause)
        sim_elapsed_s = 0.0
        rerouted = False
        recent_shortcut_edges = []  # track newly added shortcut edges for drawing

        # Draw initial marker at office
        nx.draw_networkx_nodes(G, pos, nodelist=[office], node_color="black", node_size=140, ax=ax)
        plt.pause(0.5)

        for step in range(len(full_path) - 1):
            u, v = full_path[step], full_path[step + 1]

            # Mid-trip traffic only for the first employee, once
            if emp == "Employee_1" and not traffic_triggered and step == len(full_path) // 2:
                traffic_triggered = True
                print(f"[TRAFFIC] Incident detected near node {v}. Attempting reroute via small street.")

                # Blink the traffic node
                for _ in range(TRAFFIC_BLINKS):
                    nx.draw_networkx_nodes(G, pos, nodelist=[v], node_color="red", node_size=160, ax=ax)
                    plt.title(f"{emp}: TRAFFIC at node {v}! Rerouting…")
                    plt.pause(TRAFFIC_BLINK_INTERVAL)
                    # erase blink by overdrawing node in background color
                    nx.draw_networkx_nodes(G, pos, nodelist=[v], node_color="#c0c0c0", node_size=160, ax=ax)
                    plt.pause(TRAFFIC_BLINK_INTERVAL)

                # Inject a temporary shortcut and recompute the remainder
                added = add_temporary_shortcut(G, pos, from_node=u)
                if added:
                    # record undirected edges for styling
                    for a, b in added:
                        shortcut_edges_all.add(tuple(sorted((a, b))))
                        recent_shortcut_edges.append((a, b))

                    try:
                        # Re-route from current u to dest via new shortcut, then to office
                        path_mid = nx.shortest_path(G, u, dest, weight="length")
                        path_back2 = nx.shortest_path(G, dest, office, weight="length")
                        full_path = full_path[: step + 1] + path_mid[1:] + path_back2[1:]
                        rerouted = True
                        print(f"[REROUTE] Using a temporary shortcut street. Route updated.")
                    except nx.NetworkXNoPath:
                        print("[REROUTE] No valid route found after shortcut insertion; continuing original plan.")

            # choose segment style: magenta for local/shortcut, orange otherwise
            undirected_pair = tuple(sorted((u, v)))
            seg_is_shortcut = undirected_pair in shortcut_edges_all
            seg_is_local = undirected_pair in local_set

            edge_color = "magenta" if (seg_is_shortcut or seg_is_local) else "orange"
            edge_style = "dashed" if seg_is_shortcut else "solid"
            edge_width = 2.4 if seg_is_shortcut else 2.0

            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], edge_color=edge_color, width=edge_width,
                style=edge_style, ax=ax, alpha=0.95
            )
            nx.draw_networkx_nodes(G, pos, nodelist=[u], node_color=edge_color, node_size=70, ax=ax)

            # advance simulated time for this edge (length/speed)
            L = _edge_length_m(G, u, v) or 0.0
            sim_elapsed_s += (L / AVG_SPEED_MPS)

            ax.set_title(
                f"{emp} travelling… step {step+1}/{len(full_path)-1}\n"
                f"ETA at departure: {eta_departure.strftime('%H:%M:%S')}",
                fontsize=11
            )
            plt.pause(0.25)

        # Arrived
        nx.draw_networkx_nodes(G, pos, nodelist=[office], node_color="green", node_size=180, ax=ax)
        plt.title(f"{emp}: returned to office", fontsize=12)
        plt.pause(0.8)
        plt.show(block=False)

        # Simulated ATA
        ata_time = (start_time + timedelta(seconds=sim_elapsed_s)).strftime("%H:%M:%S")
        print(f"ATA (simulated): {ata_time}")

        # Feedback
        ans = input(f"Was {emp}'s route acceptable? (yes/no): ").strip().lower()
        feedback[emp] = "positive" if ans == "yes" else "negative"

    return feedback


# ---------- Runner ----------

def run_simulation():
    G, pos = build_city_graph()
    office = choose_office(G)
    employees = assign_employees(G, num_agents=3)
    print("Employees & destinations:", employees)

    feedback = live_simulation(G, pos, employees, office)

    print("\nFinal Feedback:")
    for emp, fb in feedback.items():
        print(f"{emp}: {fb}")


if __name__ == "__main__":
    run_simulation()
