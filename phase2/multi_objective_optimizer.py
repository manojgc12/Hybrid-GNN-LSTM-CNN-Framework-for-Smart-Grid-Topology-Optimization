import copy
import pandapower as pp
import numpy as np
from radial_constraint import is_radial

def optimize_topology(base_net, model, feature_engineer):

    results = []

    for i in base_net.line[base_net.line.in_service == True].index:

        net = copy.deepcopy(base_net)
        net.line.at[i, "in_service"] = False

        try:
            pp.runpp(net)

            if net.res_bus.vm_pu.isna().any():
                continue

            if net.res_line.pl_mw.isna().any():
                continue

            # Radial constraint
            if not is_radial(net):
                continue

            # Voltage constraint
            if net.res_bus.vm_pu.min() < 0.95 or net.res_bus.vm_pu.max() > 1.05:
                continue

            graph = feature_engineer.build_graph(net)
            predicted_loss = model(graph).item()

            voltage_dev = np.sum(np.abs(net.res_bus.vm_pu - 1.0))

            # Multi-objective scoring
            score = 0.7 * predicted_loss + 0.3 * voltage_dev

            results.append((i, predicted_loss, voltage_dev, score))

        except:
            continue

    results.sort(key=lambda x: x[3])  # sort by score

    return results