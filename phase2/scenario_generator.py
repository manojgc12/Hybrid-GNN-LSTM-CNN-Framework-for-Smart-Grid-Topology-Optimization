import copy
import numpy as np
import pandapower as pp


def generate_scenarios(base_net, num_scenarios=50, variation=0.2):
    """
    Generate multiple operating scenarios by randomly scaling
    active and reactive loads within ±variation of the base case.
    Returns list of (net, total_loss_mw) tuples.
    """
    dataset = []
    np.random.seed(0)

    for _ in range(num_scenarios):
        net     = copy.deepcopy(base_net)
        p_scale = np.random.uniform(1 - variation, 1 + variation)
        q_scale = np.random.uniform(1 - variation, 1 + variation)

        net.load["p_mw"]   = net.load["p_mw"]   * p_scale
        net.load["q_mvar"] = net.load["q_mvar"] * q_scale

        try:
            pp.runpp(net, algorithm="nr", numba=False)
            loss = float(net.res_line.pl_mw.sum())
            dataset.append((net, loss))
        except Exception:
            continue

    print(f"  Generated {len(dataset)} valid scenarios "
          f"out of {num_scenarios} attempts.")
    return dataset
