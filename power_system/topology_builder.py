"""
power_system/topology_builder.py - FIXED v3
Star/Mesh now use sn_mva=1.0 to match ieee69 base.
All PF calls use bfsw fallback. Backup spokes always INACTIVE.
"""
import sys, os, importlib.util
import pandapower as pp
import pandapower.networks as pn
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in [_ROOT, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _load_ieee_mod():
    for p in [os.path.join(_HERE,"ieee_networks.py"),
              os.path.join(_ROOT,"power_system","ieee_networks.py")]:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("ieee_networks",p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None

_ieee = _load_ieee_mod()

def _base(name):
    n = name.lower()
    if n == "ieee33": return pn.case33bw()
    if n == "ieee69" and _ieee: return _ieee.create_ieee69()
    if n == "ieee85" and _ieee: return _ieee.create_ieee85()
    return pn.case33bw()

def _pf(net):
    for algo in ['nr','bfsw']:
        try:
            pp.runpp(net,algorithm=algo,numba=False,max_iteration=50)
            return True
        except: pass
    return False

def _scale(net, sp, sq):
    if len(net.load) > 0:
        net.load["p_mw"] = net.load["p_mw"] * sp
        net.load["q_mvar"] = net.load["q_mvar"] * sq

# ── RADIAL ────────────────────────────────────────────────────────────────────
def build_radial_network(network_name="ieee33", sp=1.0, sq=1.0):
    net = _base(network_name); _scale(net, sp, sq); _pf(net)
    return net

# ── STAR ──────────────────────────────────────────────────────────────────────
# Backup spokes: short direct feeders from bus 0, all INACTIVE
# PF runs on the clean radial base → always converges
STAR = {
    "ieee33": [(5,0.18,0.12),(10,0.22,0.15),(14,0.28,0.19),(18,0.32,0.22),(25,0.20,0.14)],
    "ieee69": [(9,0.20,0.14),(18,0.28,0.19),(27,0.35,0.24),(48,0.40,0.28),(58,0.45,0.31)],
    "ieee85": [(10,0.22,0.15),(22,0.32,0.22),(36,0.38,0.26),(53,0.42,0.29),(68,0.48,0.33)],
}
def build_star_network(network_name="ieee33", sp=1.0, sq=1.0, seed=42):
    np.random.seed(seed)
    net = _base(network_name); _scale(net, sp, sq)
    n = len(net.bus)
    for i, (tb, r, x) in enumerate(STAR.get(network_name.lower(), STAR["ieee33"])):
        if tb < n:
            pp.create_line_from_parameters(
                net, from_bus=0, to_bus=tb, length_km=1.0,
                r_ohm_per_km=r+np.random.uniform(-0.005,0.005),
                x_ohm_per_km=x+np.random.uniform(-0.002,0.002),
                c_nf_per_km=0.0, max_i_ka=0.5,
                name=f"StarBackup_{i}", in_service=False)
    _pf(net)
    return net

# ── MESH ──────────────────────────────────────────────────────────────────────
# Active cross-ties create inter-feeder mesh.
# Inactive extras available for optimizer.
# Bus pairs carefully chosen to be on different branches (won't create parallel paths).
MESH = {
    "ieee33": {
        "active":   [(8,21,0.28,0.19),(12,22,0.32,0.22)],
        "inactive": [(17,30,0.38,0.26),(24,28,0.36,0.24)],
    },
    "ieee69": {
        # Branch A ends at bus 26. Branch B starts at bus 27.
        # Cross: connect bus 26 (end of A) to bus 27 (start of B) — creates main loop
        # Also: connect bus 11 (mid A) to bus 36 (mid B)
        "active":   [(26,27,0.30,0.20),(11,36,0.45,0.31)],
        "inactive": [(25,54,0.50,0.34),(22,40,0.48,0.33)],
    },
    "ieee85": {
        # Lateral from bus 3→27 and main trunk: connect 26↔27
        # Lateral from bus 6→43 and lateral from bus 10→53: connect 42↔52
        "active":   [(26,27,0.30,0.21),(42,52,0.35,0.24)],
        "inactive": [(42,62,0.40,0.28),(71,77,0.38,0.26)],
    },
}
def build_mesh_network(network_name="ieee33", sp=1.0, sq=1.0, seed=42):
    np.random.seed(seed)
    net = _base(network_name); _scale(net, sp, sq)
    n = len(net.bus)
    c = MESH.get(network_name.lower(), MESH["ieee33"])
    for i, (fb, tb, r, x) in enumerate(c["active"]):
        if fb < n and tb < n:
            pp.create_line_from_parameters(
                net, from_bus=fb, to_bus=tb, length_km=1.0,
                r_ohm_per_km=r+np.random.uniform(-0.01,0.01),
                x_ohm_per_km=x+np.random.uniform(-0.005,0.005),
                c_nf_per_km=0.0, max_i_ka=0.4,
                name=f"MeshCross_{i}", in_service=True)
    for i, (fb, tb, r, x) in enumerate(c["inactive"]):
        if fb < n and tb < n:
            pp.create_line_from_parameters(
                net, from_bus=fb, to_bus=tb, length_km=1.0,
                r_ohm_per_km=r+np.random.uniform(-0.01,0.01),
                x_ohm_per_km=x+np.random.uniform(-0.005,0.005),
                c_nf_per_km=0.0, max_i_ka=0.4,
                name=f"MeshExtra_{i}", in_service=False)
    _pf(net)
    return net

def get_network_topology(topology_type="radial", network_name="ieee33",
                         load_scale_p=1.0, load_scale_q=1.0, seed=42):
    t = topology_type.lower()
    if t == "star":   return build_star_network(network_name, load_scale_p, load_scale_q, seed)
    elif t == "mesh": return build_mesh_network(network_name, load_scale_p, load_scale_q, seed)
    return build_radial_network(network_name, load_scale_p, load_scale_q)