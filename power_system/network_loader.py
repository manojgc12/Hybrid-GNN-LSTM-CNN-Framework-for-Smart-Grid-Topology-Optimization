import pandapower.networks as pn

class DistributionNetwork:
    def __init__(self, name="IEEE33"):
        self.name = name
        self.net = None

    def load_network(self):
        if self.name == "IEEE33":
            self.net = pn.case33bw()
        else:
            raise ValueError("Only IEEE33 supported in Phase-1")
        return self.net

    def summary(self):
        return {
            "buses": len(self.net.bus),
            "lines": len(self.net.line),
            "loads": len(self.net.load)
        }
