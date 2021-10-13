import numpy as np
from PIL import Image
from permutohedral import Permutohedral

im = Image.open('./lena.small.jpg')
im = np.array(im) / 255.

h, w, n_channels = im.shape

invSpatialStdev = 1. / 5.
invColorStdev = 1. / .25

features = np.zeros((5, h, w), dtype=np.float32)
spatial_feat = np.mgrid[0:h, 0:w] # y, x
spatial_feat = spatial_feat[::-1] * invSpatialStdev
color_feat = im * invColorStdev
features[:2] = spatial_feat
features[2:] = color_feat.transpose((2, 0, 1))
features = features.reshape((5, -1))

# d, N = features.shape
# lattice1 = Permutohedral(N, d)
# lattice2 = Permutohedral(N, d)

# lattice1.init(features)
# lattice2.init(features2)

# all_ones = np.ones((1, N), dtype=np.float32)

# norm1 = lattice1.compute(all_ones)
# norm1 = norm1.reshape((1, h, w))[0]
# norm2 = lattice2.compute(all_ones)
# norm2 = norm2.reshape((1, h, w))[0]

# print(np.all(norm1 == norm2))

# def hash(k):
#     print('hash')
#     r = 0
#     for i in range(len(k)):
#         r += k[i]
#         r *= 1664525
#         print('i: {}, r: {}'.format(i, r))
#     return r

# def hash2(k):
#     print('hash2')
#     r = np.zeros_like(k, dtype=np.uint64)
#     r[:] = k * 1664525 ** np.arange(len(k), 0, -1)
#     print(r)
#     return r.sum()

# k = [1, 2, 3, 4, 5]
# print(hash(k) % (2 ** 128), hash2(k) % (2 ** 128))

d, N = features.shape
scale_factor = np.zeros((d, ), dtype=np.float32)
elevated = np.zeros((d + 1, ), dtype=np.float32)

inv_std_dev = np.sqrt(2. / 3.) * (d + 1)
scale_factor[:] = 1. / np.sqrt( (np.arange(d) + 2) * (np.arange(d) + 1) )  * inv_std_dev

# My numpification
mycf = features * scale_factor[:, np.newaxis] # (d, N)
myE = np.vstack([
    np.ones((d, ), dtype=np.float32), 
    np.diag(-np.arange(d, dtype=np.float32) - 2) + np.triu(np.ones((d, d), dtype=np.float32))]) # (d + 1, d)
myelevated = np.matmul(myE, mycf) # (d, N)

for k in range(N):
    # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
    f = features[:, k]

    # sm contains the sum of 1..n of our feature vector
    sm = 0
    for j in range(d, 0, -1):
        cf = f[j - 1] * scale_factor[j - 1]
        elevated[j] = sm - j * cf
        sm += cf
    elevated[0] = sm

    if not np.allclose(elevated, myelevated[:, k]):
        print(k, elevated, myelevated[:, k])
    else:
        print('Good')