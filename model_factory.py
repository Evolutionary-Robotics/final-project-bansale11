from nn2 import FNN
from nn_drnn import DRNN
# Helper to construct either an FNN or a DRNN and load parameters from a genome
def make_net(geno, arch, layers):
    net = FNN(layers) if arch == "fnn" else DRNN(layers)
    net.set_params(geno)
    return net

# Compute length of genotype vector needed for a given architecture.
def genesize_for(arch, layers):
    if arch == "fnn":
        # weights + biases in a standard MLP
        return sum(layers[i]*layers[i+1] for i in range(len(layers)-1)) + sum(layers[1:])
    # DRNN [nin, H, nout] = Wxh + Whh + bh + Why + by
    nin, H, nout = layers
    return nin*H + H*H + H + H*nout + nout
