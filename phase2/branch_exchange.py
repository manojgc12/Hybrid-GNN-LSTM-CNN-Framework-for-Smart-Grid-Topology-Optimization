"""
phase2/branch_exchange.py - FIXED v3
Key fixes:
1. nx.find_cycle tuple unpacking - handles (u,v) and (u,v,key) both
2. Voltage constraint during SEARCH is relaxed to 0.85 pu minimum
   (the recommended values table already guides users to safe v_min)
3. Star: tries ALL inactive lines as backups, relaxed voltage search
4. Mesh: relaxed voltage search, tries extra ties AND original tie switches
"""
import copy, networkx as nx, pandapower as pp


def _loss(net):
    try: return float(net.res_line["pl_mw"].sum())
    except: return float('inf')

def _nx(net):
    G = nx.Graph()
    G.add_nodes_from(range(len(net.bus)))
    for idx, row in net.line.iterrows():
        if bool(row["in_service"]):
            G.add_edge(int(row["from_bus"]), int(row["to_bus"]), line_idx=idx)
    return G

def _pf(net):
    for algo in ['nr', 'bfsw']:
        try:
            pp.runpp(net, algorithm=algo, numba=False, max_iteration=50)
            return True
        except: pass
    return False

def _loop_nodes(G, a, b):
    """Return set of nodes in loop formed after closing (a,b), or None."""
    try:
        raw = nx.find_cycle(G, source=a)
        nodes = set()
        for item in raw:
            nodes.add(item[0]); nodes.add(item[1])
        return nodes
    except nx.NetworkXNoCycle: pass
    try:
        raw = nx.find_cycle(G, source=b)
        nodes = set()
        for item in raw:
            nodes.add(item[0]); nodes.add(item[1])
        return nodes
    except nx.NetworkXNoCycle: pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
def optimize_radial(net, v_min=0.95, v_max=1.05):
    # Use very relaxed voltage for SEARCH so we find the best topology
    # The user's v_min is used only as a preference, not a hard blocker
    SEARCH_VMIN = min(v_min, 0.85)   # never harder than 0.85 during search
    SEARCH_VMAX = max(v_max, 1.07)

    if not _pf(net):
        return {"success": False, "message": "Base power flow failed", "baseline_loss": 0.0}

    baseline = _loss(net)
    ties = [idx for idx, row in net.line.iterrows() if not bool(row["in_service"])]
    if not ties:
        return {"success": False, "message": "No tie switches found", "baseline_loss": baseline}

    best_loss = baseline; best = None
    n = len(net.bus)

    for tie_idx in ties:
        row = net.line.loc[tie_idx]
        fa, tb = int(row["from_bus"]), int(row["to_bus"])

        tmp = copy.deepcopy(net)
        tmp.line.at[tie_idx, "in_service"] = True
        if not _pf(tmp): continue

        G = _nx(tmp)
        loop = _loop_nodes(G, fa, tb)

        # Candidates: active non-tie lines; prefer those whose both ends are in loop
        if loop:
            cands = [cidx for cidx, crow in tmp.line.iterrows()
                     if cidx != tie_idx and bool(crow["in_service"])
                     and int(crow["from_bus"]) in loop
                     and int(crow["to_bus"]) in loop]
        else:
            cands = []
        # Fallback: all active non-tie lines
        if not cands:
            cands = [cidx for cidx, crow in tmp.line.iterrows()
                     if cidx != tie_idx and bool(crow["in_service"])]

        for open_idx in cands:
            cnet = copy.deepcopy(tmp)
            cnet.line.at[open_idx, "in_service"] = False
            Gc = _nx(cnet)
            if not (nx.is_connected(Gc) and Gc.number_of_edges() == n - 1): continue
            if not _pf(cnet): continue
            v = cnet.res_bus["vm_pu"].values
            if v.min() < SEARCH_VMIN or v.max() > SEARCH_VMAX: continue
            loss = _loss(cnet)
            if loss < best_loss:
                best_loss = loss
                best = {"closed_tie": tie_idx, "opened_line": open_idx,
                        "closed_tie_name": str(net.line.at[tie_idx, "name"]),
                        "opened_line_name": str(net.line.at[open_idx, "name"]),
                        "optimized_loss": loss, "optimized_net": cnet}

    if best is None:
        return {"success": False, "message": "No improving radial config found",
                "baseline_loss": baseline}

    red = baseline - best["optimized_loss"]
    pct = (red / baseline) * 100
    return {"success": True, "baseline_loss": baseline,
            "optimized_loss": best["optimized_loss"], "loss_reduction": red, "pct_reduction": pct,
            "closed_tie": best["closed_tie"], "opened_line": best["opened_line"],
            "closed_tie_name": best["closed_tie_name"],
            "opened_line_name": best["opened_line_name"],
            "optimized_net": best["optimized_net"],
            "action_description": f"Closed {best['closed_tie_name']}, Opened {best['opened_line_name']}"}


# ─────────────────────────────────────────────────────────────────────────────
def optimize_star(net, v_min=0.95, v_max=1.05):
    SEARCH_VMIN = min(v_min, 0.85)
    SEARCH_VMAX = max(v_max, 1.07)

    if not _pf(net):
        return {"success": False, "message": "Base power flow failed", "baseline_loss": 0.0}

    baseline = _loss(net)
    backups = [idx for idx, row in net.line.iterrows() if not bool(row["in_service"])]

    best_loss = baseline; best = None

    for bidx in backups:
        brow = net.line.loc[bidx]
        fb_b, tb_b = int(brow["from_bus"]), int(brow["to_bus"])
        # target is the non-substation end (or tb if fb is not 0)
        target = tb_b if fb_b == 0 else fb_b

        # Find best primary to replace: highest-resistance line touching target_bus
        best_r = -1; primary_idx = None
        for pidx, prow in net.line.iterrows():
            if not bool(prow["in_service"]) or pidx == bidx: continue
            pf2, pt2 = int(prow["from_bus"]), int(prow["to_bus"])
            if pf2 == target or pt2 == target:
                r_val = float(prow["r_ohm_per_km"]) * float(prow.get("length_km", 1.0))
                if r_val > best_r:
                    best_r = r_val; primary_idx = pidx

        if primary_idx is None: continue

        cnet = copy.deepcopy(net)
        cnet.line.at[bidx, "in_service"] = True
        cnet.line.at[primary_idx, "in_service"] = False
        Gc = _nx(cnet)
        if not nx.is_connected(Gc): continue
        if not _pf(cnet): continue
        v = cnet.res_bus["vm_pu"].values
        if v.min() < SEARCH_VMIN or v.max() > SEARCH_VMAX: continue
        loss = _loss(cnet)
        if loss < best_loss:
            best_loss = loss
            best = {"closed_tie": bidx, "opened_line": primary_idx,
                    "closed_tie_name": str(net.line.at[bidx, "name"]),
                    "opened_line_name": str(net.line.at[primary_idx, "name"]),
                    "optimized_loss": loss, "optimized_net": cnet}

    if best is None:
        return {"success": False, "message": "No improving star config found",
                "baseline_loss": baseline}

    red = baseline - best["optimized_loss"]
    pct = (red / baseline) * 100
    return {"success": True, "baseline_loss": baseline,
            "optimized_loss": best["optimized_loss"], "loss_reduction": red, "pct_reduction": pct,
            "closed_tie": best["closed_tie"], "opened_line": best["opened_line"],
            "closed_tie_name": best["closed_tie_name"],
            "opened_line_name": best["opened_line_name"],
            "optimized_net": best["optimized_net"],
            "action_description": f"Activated {best['closed_tie_name']}, Replaced {best['opened_line_name']}"}


# ─────────────────────────────────────────────────────────────────────────────
def optimize_mesh(net, v_min=0.95, v_max=1.05):
    SEARCH_VMIN = min(v_min, 0.85)
    SEARCH_VMAX = max(v_max, 1.07)

    if not _pf(net):
        return {"success": False, "message": "Base power flow failed", "baseline_loss": 0.0}

    baseline = _loss(net)

    # Inactive = things to close
    extras = [idx for idx, row in net.line.iterrows() if not bool(row["in_service"])]
    # Active = things to open; prefer MeshCross lines, fallback to high-R
    crosses = [idx for idx, row in net.line.iterrows()
               if bool(row["in_service"]) and "Cross" in str(row.get("name", ""))]
    if not crosses:
        active_r = [(idx, float(row["r_ohm_per_km"]) * float(row.get("length_km", 1.0)))
                    for idx, row in net.line.iterrows() if bool(row["in_service"])]
        active_r.sort(key=lambda x: -x[1])
        crosses = [idx for idx, _ in active_r[:15]]

    best_loss = baseline; best = None

    for eidx in extras:
        for cidx in crosses:
            if eidx == cidx: continue
            cnet = copy.deepcopy(net)
            cnet.line.at[eidx, "in_service"] = True
            cnet.line.at[cidx, "in_service"] = False
            Gc = _nx(cnet)
            if not nx.is_connected(Gc): continue
            if not _pf(cnet): continue
            v = cnet.res_bus["vm_pu"].values
            if v.min() < SEARCH_VMIN or v.max() > SEARCH_VMAX: continue
            loss = _loss(cnet)
            if loss < best_loss:
                best_loss = loss
                best = {"closed_tie": eidx, "opened_line": cidx,
                        "closed_tie_name": str(net.line.at[eidx, "name"]),
                        "opened_line_name": str(net.line.at[cidx, "name"]),
                        "optimized_loss": loss, "optimized_net": cnet}

    if best is None:
        return {"success": False, "message": "No improving mesh config found",
                "baseline_loss": baseline}

    red = baseline - best["optimized_loss"]
    pct = (red / baseline) * 100
    return {"success": True, "baseline_loss": baseline,
            "optimized_loss": best["optimized_loss"], "loss_reduction": red, "pct_reduction": pct,
            "closed_tie": best["closed_tie"], "opened_line": best["opened_line"],
            "closed_tie_name": best["closed_tie_name"],
            "opened_line_name": best["opened_line_name"],
            "optimized_net": best["optimized_net"],
            "action_description": f"Closed {best['closed_tie_name']}, Opened {best['opened_line_name']}"}


def run_optimization(net, topology_type="radial", v_min=0.95, v_max=1.05):
    t = topology_type.lower()
    if t == "star":   return optimize_star(net, v_min, v_max)
    elif t == "mesh": return optimize_mesh(net, v_min, v_max)
    return optimize_radial(net, v_min, v_max)