''' This is a python implementation of permutohedral lattice in Dense CRF
'''

import numpy as np

class HashTable:
    def __init__(self, key_size, n_elements):
        self.key_size_ = key_size
        self.filled_ = 0
        self.capacity_ = 2 * n_elements
        self.keys_ = np.zeros( ((self.capacity_ // 2 + 10) * self.key_size_, ), dtype=np.short)
        self.table_ = np.ones((2 * n_elements, ), dtype=np.int32) * -1

    def grow(self):
        # Create the new memory and copy the values in
        old_capacity = self.capacity_
        self.capacity_ *= 2
        old_keys = np.zeros(((old_capacity + 10) * self.key_size_, ), dtype=np.short)
        old_keys[:(old_capacity // 2 + 10) * self.key_size_] = self.keys_
        old_table = np.ones((self.capacity_, ), dtype=np.int32) * -1

        # Swap the memory
        self.table_, old_table = old_table, self.table_
        self.keys_, old_keys = old_keys, self.keys_

        # Reinsert each element
        for i in range(old_capacity):
            if old_table[i] >= 0:
                e = old_table[i]
                h = self.hash(self.get_key(e)) % self.capacity_
                while self.table_[h] >= 0:
                    if h < self.capacity_ - 1:
                        h = h + 1
                    else:
                        h = 0
                self.table_[h] = e

    def hash(self, k):
        r = np.int64(0)
        for i in range(self.key_size_):
            r += k[i]
            r *= 1664525
        return r

    def size(self):
        return self.filled_

    def reset(self):
        self.filled_ = 0
        self.table_.fill(-1)

    def find(self, k, create=False):
        if self.capacity_ <= 2 * self.filled_:
            self.grow()
        # Get the hash value
        h = self.hash(k) % self.capacity_
        # Find the element with the right key, using linear probing
        while True:
            e = self.table_[h]
            if e == -1:
                if create:
                    # Insert a new key and return the new id
                    self.keys_[self.filled_ * self.key_size_:self.filled_ * self.key_size_ + self.key_size_] = k[:self.key_size_]
                    self.table_[h] = self.filled_
                    self.filled_ += 1
                    return self.table_[h]
                else:
                    return -1
            # Check if the current key is The One
            good = np.all(self.keys_[e * self.key_size_:e * self.key_size_ + self.key_size_] == k[:self.key_size_])
            if good:
                return e
            # Continue searching
            h += 1
            if h == self.capacity_:
                h = 0

    def get_key(self, i):
        return self.keys_[i * self.key_size_:]

class Permutohedral:
    def __init__(self, N, d):
        self.N_, self.M_, self.d_ = N, 0, d
        self.offset_ = np.zeros( ((d + 1) * N, ), dtype=np.int32)
        self.rank_ = np.zeros( ((d + 1) * N, ), dtype=np.int32)
        self.barycentric_ = np.zeros( ((d + 1) * N, ), dtype=np.float32)
        self.blur_neighbors_ = None
        self.hash_table = HashTable(d, N * (d + 1))

    def init(self, feature):
        # Compute the lattice coordinates for each feature [there is going to be a lot of magic here
        # pass

        # Allocate the class memory
        # pass

        # Allocate the local memory
        scale_factor = np.zeros((self.d_, ), dtype=np.float32)
        elevated = np.zeros((self.d_ + 1, ), dtype=np.float32)
        rem0 = np.zeros((self.d_ + 1, ), dtype=np.float32)
        barycentric = np.zeros((self.d_ + 2, ), dtype=np.float32)
        rank = np.zeros((self.d_ + 1, ), dtype=np.short)
        canonical = np.zeros( ((self.d_ + 1) * (self.d_ + 1), ), dtype=np.short)
        key = np.zeros((self.d_ + 1, ), dtype=np.short)

        # Compute the canonical simplex
        for i in range(self.d_ + 1):
            for j in range(self.d_ - i + 1):
                canonical[i * (self.d_ + 1) + j] = i
            for j in range(self.d_ - i + 1, self.d_ + 1):
                canonical[i * (self.d_ + 1) + j] = i - (self.d_ + 1)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2. / 3.) * (self.d_ + 1)
        # Compute the diagonal part of E (p.5 in [Adams et al 2021])
        for i in range(self.d_):
            scale_factor[i] = 1. / np.sqrt( np.double((i + 2) * (i + 1)) ) * inv_std_dev

        # Compute the simplex each feature lies in
        for k in range(self.N_):
            # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
            f = feature[:, k]

            # sm contains the sum of 1..n of our feature vector
            sm = 0
            for j in range(self.d_, 0, -1):
                cf = f[j - 1] * scale_factor[j - 1]
                elevated[j] = sm - j * cf
                sm += cf
            elevated[0] = sm

            # Find the closest 0-colored simplex through rounding
            down_factor = 1. / (self.d_ + 1)
            up_factor = self.d_ + 1
            _sum = 0
            for i in range(self.d_ + 1):
                v = down_factor * elevated[i]
                up = np.ceil(v) * up_factor
                down = np.floor(v) * up_factor
                if (up - elevated[i] < elevated[i] - down):
                    rd2 = np.short(up)
                else:
                    rd2 = np.short(down)

                rem0[i] = rd2
                _sum += rd2 * down_factor

            # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
            rank.fill(0)
            for i in range(self.d_):
                di = elevated[i] - rem0[i]
                for j in range(i + 1, self.d_ + 1):
                    if di < elevated[j] - rem0[j]:
                        rank[i] += 1
                    else:
                        rank[j] += 1

            # If the point doesn't lie on the plane (sum != 0) bring it back
            for i in range(self.d_ + 1):
                rank[i] += _sum
                if rank[i] < 0:
                    rank[i] += self.d_ + 1
                    rem0[i] += self.d_ + 1
                elif rank[i] > self.d_:
                    rank[i] -= self.d_ + 1
                    rem0[i] -= self.d_ + 1

            # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
            barycentric.fill(0)
            for i in range(self.d_ + 1):
                v = (elevated[i] - rem0[i]) * down_factor
                barycentric[self.d_ - rank[i]] += v
                barycentric[self.d_ - rank[i] + 1] -= v
            # Wrap around
            barycentric[0] += 1. + barycentric[self.d_ + 1]

            # Compute all vertices and their offset
            for remainder in range(self.d_ + 1):
                for i in range(self.d_):
                    key[i] = rem0[i] + canonical[remainder * (self.d_ + 1) + rank[i]]
                self.offset_[k * (self.d_ + 1) + remainder] = self.hash_table.find(key, True)
                self.rank_[k * (self.d_ + 1) + remainder] = rank[remainder]
                self.barycentric_[k * (self.d_ + 1) + remainder] = barycentric[remainder]

        del scale_factor, elevated, rem0, barycentric, rank, canonical, key

        # Find the neighbors of each lattice point

        # Get the number of vertices in the lattice
        self.M_ = self.hash_table.size()

        # Create the neighborhood structure
        self.blur_neighbors_ = np.zeros( ((self.d_ + 1) * self.M_, 2), dtype=np.int32)
        
        n1 = np.zeros((self.d_ + 1, ), dtype=np.short)
        n2 = np.zeros((self.d_ + 1, ), dtype=np.short)

        # For each of d+1 axes,
        for j in range(self.d_ + 1):
            for i in range(self.M_):
                key = self.hash_table.get_key(i)
                for k in range(self.d_):
                    n1[k] = key[k] - 1
                    n2[k] = key[k] + 1

                n1[j] = key[j] + self.d_
                n2[j] = key[j] - self.d_

                self.blur_neighbors_[j * self.M_ + i, 0] = self.hash_table.find(n1)
                self.blur_neighbors_[j * self.M_ + i, 1] = self.hash_table.find(n2)
        
        del n1, n2

    def seq_compute(self, out, inp, value_size, reverse):
        # Shift all values by 1 such that -1 -> 0 (used for blurring)
        values = np.zeros( ((self.M_ + 2) * value_size, ), dtype=np.float32)
        new_values = np.zeros( ((self.M_ + 2) * value_size, ), dtype=np.float32)

        # Splatting
        for i in range(self.N_):
            for j in range(self.d_ + 1):
                o = self.offset_[i * (self.d_ + 1) + j] + 1
                w = self.barycentric_[i * (self.d_ + 1) + j]
                for k in range(value_size):
                    values[o * value_size + k] += w * inp[i * value_size + k]

        j_range = range(self.d_, -1, -1) if reverse else range(self.d_ + 1)
        for j in j_range:
            for i in range(self.M_):
                old_val = values[(i + 1) * value_size:(i + 2) * value_size]
                new_val = new_values[(i + 1) * value_size:(i + 2) * value_size]

                n1 = self.blur_neighbors_[j * self.M_ + i, 0] + 1
                n2 = self.blur_neighbors_[j * self.M_ + i, 1] + 1
                n1_val = values[n1 * value_size:n1 * value_size + value_size]
                n2_val = values[n2 * value_size:n2 * value_size + value_size]

                for k in range(value_size):
                    new_val[k] = old_val[k] + .5 * (n1_val[k] + n2_val[k])

            values, new_values = new_values, values

        # Alpha is a magic scaling constant (write Andrew if you reall wanna understand this)
        alpha = 1. / (1 + np.power(2., -self.d_))
        
        # Slicing
        for i in range(self.N_):
            for k in range(value_size):
                out[i * value_size + k] = 0
            for j in range(self.d_ + 1):
                o = self.offset_[i * (self.d_ + 1) + j] + 1
                w = self.barycentric_[i * (self.d_ + 1) + j]
                for k in range(value_size):
                    out[i * value_size + k] += w * values[o * value_size + k] * alpha

        del values, new_values

    def compute(self, inp, reverse=False):
        vs, col = inp.shape
        inp = inp.flatten()
        out = np.zeros_like(inp)
        self.seq_compute(out, inp, vs, reverse)
        out = out.reshape((vs, col))
        return out