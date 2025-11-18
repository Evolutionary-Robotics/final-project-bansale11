import numpy as np

class DRNN:
    def __init__(self, layers, w_scale=4.0, b_scale=4.0):
        # Exactly three entries: input size, hidden size, output size
        assert len(layers) == 3, "Use layers [nin, H, nout] for DRNN"
        self.layers = layers
        self.nin, self.H, self.nout = layers
        self.w_scale, self.b_scale = w_scale, b_scale
        self.h = np.zeros((1, self.H), dtype=float)
        self.Wxh = self.Whh = self.bh = self.Why = self.by = None

    def reset(self):
        self.h[:] = 0.0

    def set_params(self, g):
        p = 0
        Wxh_n = self.nin * self.H
        self.Wxh = g[p:p+Wxh_n].reshape(self.nin, self.H) * self.w_scale; p += Wxh_n
        Whh_n = self.H * self.H
        self.Whh = g[p:p+Whh_n].reshape(self.H, self.H) * (self.w_scale/np.sqrt(self.H)); p += Whh_n
        self.bh  = g[p:p+self.H].reshape(1, self.H) * self.b_scale; p += self.H
        Why_n = self.H * self.nout
        self.Why = g[p:p+Why_n].reshape(self.H, self.nout) * self.w_scale; p += Why_n
        self.by  = g[p:p+self.nout].reshape(1, self.nout) * self.b_scale; p += self.nout
        assert p == len(g), "genotype length mismatch for DRNN"

    def forward(self, x):
        x = np.asarray(x).reshape(1, -1)
        self.h = np.tanh(x @ self.Wxh + self.h @ self.Whh + self.bh)
        y = self.h @ self.Why + self.by
        return y
