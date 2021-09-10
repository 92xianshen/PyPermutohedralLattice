import numpy as np
from PIL import Image
from permutohedral import Permutohedral

im = Image.open('./lena.small.jpg')
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

features = np.zeros((5, h, w), dtype=np.float32)
for r in range(h):
    for c in range(w):
        features[0, r, c] = invSpatialStdev * c
        features[1, r, c] = invSpatialStdev * r
        features[2, r, c] = invColorStdev * im[r, c, 0]
        features[3, r, c] = invColorStdev * im[r, c, 1]
        features[4, r, c] = invColorStdev * im[r, c, 2]
features = features.reshape((5, -1))

features2 = np.zeros((5, h, w), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w]
spatial_feat = spatial_feat[::-1] * invSpatialStdev
color_feat = im * invColorStdev
features2[:2] = spatial_feat
features2[2:] = color_feat.transpose((2, 0, 1))
features2 = features2.reshape((5, -1))

d, N = features.shape
lattice1 = Permutohedral(N, d)
lattice2 = Permutohedral(N, d)

lattice1.init(features)
lattice2.init(features2)

all_ones = np.ones((1, N), dtype=np.float32)

norm1 = lattice1.compute(all_ones)
norm1 = norm1.reshape((1, h, w))[0]
norm2 = lattice2.compute(all_ones)
norm2 = norm2.reshape((1, h, w))[0]

print(np.all(norm1 == norm2))
