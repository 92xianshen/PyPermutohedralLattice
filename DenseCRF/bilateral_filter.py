import numpy as np
from PIL import Image
from permutohedral import Permutohedral
import matplotlib.pyplot as plt
import cv2

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
# features = np.zeros((2, h, w), dtype=np.float32)
# for r in range(h):
#     for c in range(w):
#         features[0, r, c] = invSpatialStdev * c
#         features[1, r, c] = invSpatialStdev * r
# features = features.reshape((2, -1))

N, d = features.shape[1], features.shape[0]
lattice = Permutohedral(N, d)
lattice.init(features)

all_ones = np.ones((1, N), dtype=np.float32)
all_ones = lattice.compute(all_ones)
all_ones = all_ones.reshape((1, h, w))[0]

im_filtered = np.zeros_like(im)
for ch in range(n_channels):
    imch = im[..., ch:ch + 1].transpose((2, 0, 1)).reshape((1, -1))
    imch_filtered = lattice.compute(imch)
    imch_filtered = imch_filtered.reshape((1, h, w))[0]
    imch_filtered = imch_filtered / all_ones
    imch_filtered = (imch_filtered - imch_filtered.min()) / (imch_filtered.max() - imch_filtered.min())
    im_filtered[..., ch] = imch_filtered

cv2.imshow('im', im[..., ::-1])
cv2.imshow('im_filtered', im_filtered[..., ::-1])
cv2.waitKey()

# im_add = im.transpose((2, 0, 1)).reshape((n_channels, -1))
# im_add = np.vstack([im_add, np.ones((1, h * w), dtype=im.dtype)])
# print(im_add.shape)

# im_filtered = lattice.compute(im_add)
# im_filtered = (im_filtered[:3] / im_filtered[-1:])
# print(im_filtered.max(), im_filtered.min())
# im_filtered = im_filtered.reshape((n_channels, h, w)).transpose((1, 2, 0))
# plt.imshow(im_filtered / im_filtered.max())
# plt.show()

# # all_ones = np.ones((1, N), dtype=np.float32)
# # all_ones = lattice.compute(all_ones)
# # all_ones = all_ones.reshape((1, h, w))[0]

# # plt.imshow(all_ones / all_ones.max(), cmap='gray')
# # plt.show()