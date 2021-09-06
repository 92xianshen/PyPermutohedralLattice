import numpy as np

d = 5
d1 = d + 1
position = np.array([.5, .5, .5, .5, .5])
pos_idx = 0

# # ->> 1. PermutohedralLattice.__init__.canonical
# # Original
# d1 = 5 + 1
# canonical = np.zeros((d1 ** 2), dtype='int16')
# for i in range(d1):
#     for j in range(d1 - i):
#         canonical[i * d1 + j] = i
#     for j in range(d1 - i, d1):
#         canonical[i * d1 + j] = i - d1

# print(canonical)

# canonical_np = np.zeros((d1, d1), dtype='int16')
# for i in range(d1):
#     canonical_np[i, :d1 - i] = i
#     canonical_np[i, d1 - i:] = i - d1
# canonical_np = canonical_np.flatten()

# # canonical_np[:, :d1 - np.arange(d1, dtype=np.int32)] = 1
# print(canonical_np)
# print(np.all(canonical_np == canonical))


# # ->> 2. PermutohedralLattice.__init__.scale_factor
# scale_factor = np.zeros((d), dtype='float32')
# expected_std = d1 * np.sqrt(2 / 3.)
# # Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
# for i in range(d):
#     # the diagonal entries for normalization
#     scale_factor[i] = expected_std / np.sqrt((i + 1) * (i + 2))
# print(scale_factor)

# scale_factor = np.zeros((d, ), dtype='float32')
# expected_std = d1 * np.sqrt(2 / 3.)
# scale_factor[:] = expected_std / np.sqrt((np.arange(d) + 1) * (np.arange(d) + 2))
# print(scale_factor)

# # # ->> 3. PermutohedralLattice.splat.elevated
# # # first rotate position into the (d+1)-dimensional hyperplane
# elevated = np.zeros(d1, dtype='float32')
# # scale_factor = np.ones((d, ), dtype='float32')
# # position = np.ones((d, ), dtype='float32')
# pos_idx = 0
# elevated[d] = -d * position[pos_idx + d - 1] * scale_factor[d - 1]
# for i in range(d - 1, 0, -1):
#     elevated[i] = elevated[i + 1] - i * position[pos_idx + i - 1] * scale_factor[i - 1] + (i + 2) * position[pos_idx + i] * scale_factor[i]
# elevated[0] = elevated[1] + 2 * position[pos_idx] * scale_factor[0]
# print(elevated)

# E = np.vstack([np.ones((d, ), dtype='float32'), np.diag(-np.arange(d, dtype='float32') - 2) + np.triu(np.ones((d, d), dtype='float32'))])
# scale_factor2 = np.diag(scale_factor)
# elevated2 = np.dot(np.dot(E, scale_factor2), position[:, np.newaxis]).flatten()
# print(elevated2)


# ->> 4. PermutohedralLattice.splat.barycentric
# reset barycentric
# barycentric *= 0
# t = (elevated - greedy) * splat_scale
# # Compute barycentric coordinates (See pg.10 of paper.)
# for i in range(d1):
#     barycentric[d - rank[i]] += t[i]
#     barycentric[d1 - rank[i]] -= t[i]

# barycentric[0] += 1. + barycentric[d1]

# barycentric[d - rank[np.arange(d1)]] += t[np.arange[d1]]
# barycentric[d]

# ->> 5. PermutohedralLattice.splat.el_minus_gr
el_minus_gr = [1, 3, 3, 5, 4]
rank = [0, 0, 0, 0, 0]
for i in range(4):
    for j in range(i + 1, 5):
        if el_minus_gr[i] < el_minus_gr[j]:
            rank[i] += 1
        else:
            rank[j] += 1
print(rank)
# print(np.argsort(el_minus_gr))
print(4 - np.argsort(el_minus_gr, kind='quicksort'))