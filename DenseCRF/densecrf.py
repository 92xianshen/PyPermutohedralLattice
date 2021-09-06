import numpy as np
from unary import UnaryEnergy

class DenseCRF:
    def __init__(self, N, M):
        self.N_ = N
        self.M_ = M
        self.unary_ = UnaryEnergy()

    def add_pairwise_energy(self, features, label_compatibility_function=None, kernel_type=None, normalization_type=None):
        pass

    def add_pairwise_gaussian(self, sx, sy, h, w):
        feature = np.zeros((2, self.N_), dtype=np.float32)
        for j in range(h):
            for i in range(w):
                feature[0, j * w + i] = i / sx
                feature[1, j * w + i] = j / sy

        self.add_pairwise_energy(feature)

    def add_pairwise_bilateral(self, sx, sy, sr, sg, sb, im):
        feature = np.zeros((5, self.N_), dtype=np.float32)
        h, w = im.shape[0], im.shape[1]
        for j in range(h):
            for i in range(w):
                feature[0, j * w + i] = i / sx
                feature[1, j * w + i] = j / sy
                feature[2, j * w + i] = im[j, i, 0] / sr
                feature[3, j * w + i] = im[j, i, 1] / sg
                feature[4, j * w + i] = im[j, i, 2] / sb

        self.add_pairwise_energy(feature)

    def inference(self, n_iteration):
        def exp_normalize(inp):
            out = inp - inp.max(axis=0)
            out = np.exp(out)
            out = out / out.sum(axis=0)
            return out
        Q = np.zeros((self.M_, self.N_), dtype=np.float32)
        unary = np.zeros((self.M_, self.N_), dtype=np.float32)

        Q = exp_normalize(-unary)

        for _ in range(n_iteration):
            tmp1 = -unary

            # Bilateral
            

