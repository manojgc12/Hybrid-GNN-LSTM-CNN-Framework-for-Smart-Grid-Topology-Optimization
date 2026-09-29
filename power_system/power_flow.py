"""
power_system/power_flow.py  —  FIXED with robust convergence
Runs pandapower power-flow with multiple fallback strategies.
"""
import pandapower as pp
import numpy as np


def run_power_flow(net):
    """
    Run power flow with robust fallback strategies.
    Returns dict with success flag and metrics.
    Tries: NR (flat start) → BFSW → NR (high tol) → return error gracefully.
    """
    try:
        # Try 1: Newton-Raphson with flat start and tight tolerance
        pp.runpp(net, algorithm="nr", numba=False, init="flat",
                 tolerance_mva=1e-3, max_iteration=100, verbose=False)
        success = True
    except Exception as e1:
        try:
            # Try 2: Backward-Forward Sweep (robust for radial networks)
            pp.runpp(net, algorithm="bfsw", numba=False, max_iteration=100, verbose=False)
            success = True
        except Exception as e2:
            try:
                # Try 3: NR with very loose tolerance
                pp.runpp(net, algorithm="nr", numba=False, init="flat",
                         tolerance_mva=1e-2, max_iteration=200, verbose=False)
                success = True
            except Exception as e3:
                return {
                    "success": False,
                    "error": f"PF failed: NR={str(e1)[:60]}, BFSW={str(e2)[:60]}, NR-loose={str(e3)[:60]}"
                }

    if not success or not hasattr(net, "res_bus") or len(net.res_bus) == 0:
        return {"success": False, "error": "Power flow did not produce results"}

    try:
        vm = net.res_bus["vm_pu"].values
        line_loading = net.res_line["loading_percent"].values if len(net.res_line) > 0 else np.array([0.0])
        p_loss = net.res_line["pl_mw"].sum() if len(net.res_line) > 0 else 0.0
        q_loss = net.res_line["ql_mvar"].sum() if len(net.res_line) > 0 else 0.0

        return {
            "success": True,
            "v_min": float(vm.min()),
            "v_max": float(vm.max()),
            "v_mean": float(vm.mean()),
            "p_loss_mw": float(p_loss),
            "q_loss_mvar": float(q_loss),
            "max_line_loading": float(line_loading.max()),
            "vm_pu": vm.tolist(),
            "line_loading": line_loading.tolist(),
        }
    except Exception as e:
        return {"success": False, "error": f"Metrics extraction failed: {e}"}


def scale_loads(net, active_pct: float, reactive_pct: float):
    """Scale all loads by given percentages (100 = nominal)."""
    import copy
    net2 = copy.deepcopy(net)
    net2.load["p_mw"] = net2.load["p_mw"] * (active_pct / 100.0)
    net2.load["q_mvar"] = net2.load["q_mvar"] * (reactive_pct / 100.0)
    return net2