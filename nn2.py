# Simple MLP tanh hidden
import numpy as np

class FNN:
    def __init__(self, layers, w_scale=4.0, b_scale=4.0):
        self.layers = layers
        self.L = len(layers)
        self.w_scale = w_scale
        self.b_scale = b_scale
        self.W, self.b = [], [] # Lists of weight matrices and bias vectors

    def set_params(self, g):
        self.W, self.b = [], []
        p = 0
        for l in range(self.L-1):
            ni, no = self.layers[l], self.layers[l+1]
            wn, bn = ni*no, no
            W = g[p:p+wn].reshape(ni, no) * self.w_scale; p += wn
            b = g[p:p+bn].reshape(1, no) * self.b_scale; p += bn
            self.W.append(W); self.b.append(b)

    @staticmethod
    def _tanh(x): return np.tanh(x)

    def forward(self, x):
        z = np.asarray(x);  z = z[None, :] if z.ndim == 1 else z
        for l in range(self.L-2):
            z = self._tanh(z @ self.W[l] + self.b[l])
        z = z @ self.W[-1] + self.b[-1]
        return z
