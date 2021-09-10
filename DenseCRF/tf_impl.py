''' This is a tensorflow implementation of permutohedral lattice in Dense CRF
'''

import numpy as np
import tensorflow as tf

class HashTable:
    def __init__(self, key_size, n_elements):
        self.key_size_ = key_size
        self.filled_ = 0
        self.capacity_ = 2 * n_elements
        self.keys_ = tf.Variable(
            np.zeros( 
                ((self.capacity_ // 2 + 10) * self.key_size_, ), dtype=np.short))
        self.table_ = tf.Variable(
            np.ones(
                (2 * n_elements, ), dtype=np.int32) * -1)
        self.r = tf.Variable(0, dtype=tf.int64)

    def grow(self):
        # Create the new memory and copy the values in
        print('grow...')
        old_capacity = self.capacity_
        self.capacity_ *= 2
        old_keys = tf.Variable(
            np.zeros(
                ((old_capacity + 10) * self.key_size_, ), dtype=np.short))
        old_keys[:(old_capacity // 2 + 10) * self.key_size_].assign(self.keys_)
        old_table = tf.Variable(
            np.ones(
                (self.capacity_, ), dtype=np.int32) * -1)

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
                self.table_[h].assign(e)

    def hash(self, k):
        self.r.assign(0)
        for i in range(self.key_size_):
            self.r.assign_add(tf.cast(k[i], dtype=tf.int64))
            self.r.assign(self.r * 1664525)
        return self.r

    def size(self):
        return self.filled_

    def reset(self):
        self.filled_ = 0
        self.table_.assign(tf.ones_like(self.table_) * -1)

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
                    self.keys_[self.filled_ * self.key_size_:self.filled_ * self.key_size_ + self.key_size_].assign(k[:self.key_size_])
                    self.table_[h].assign(self.filled_)
                    self.filled_ += 1
                    return self.table_[h]
                else:
                    return -1
            # Check if the current key is The One
            good = tf.reduce_all(self.keys_[e * self.key_size_:e * self.key_size_ + self.key_size_] == k[:self.key_size_])
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
        # self.offset_ = np.zeros( ((d + 1) * N, ), dtype=np.int32)
        # self.rank_ = np.zeros( ((d + 1) * N, ), dtype=np.int32)
        # self.barycentric_ = np.zeros( ((d + 1) * N, ), dtype=np.float32)
        self.blur_neighbors_ = None
        self.hash_table = HashTable(d, N * (d + 1))
        # ->> Numpify
        self.offset_ = tf.Variable(
            np.zeros( (N, (d + 1)), dtype=np.int32 ))
        self.rank_ = tf.Variable(
            np.zeros( (N, (d + 1)), dtype=np.int32 ))
        self.barycentric_ = tf.Variable(
            np.zeros( (N, (d + 1)), dtype=np.float32 ))

    def init(self, feature):
        # Compute the lattice coordinates for each feature [there is going to be a lot of magic here
        # pass

        # Allocate the class memory
        # pass

        # Allocate the local memory
        scale_factor = tf.Variable(
            np.zeros((self.d_, ), dtype=np.float32))
        elevated = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.float32))
        rem0 = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.float32))
        barycentric = tf.Variable(
            np.zeros((self.d_ + 2, ), dtype=np.float32))
        rank = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.int32))
        canonical = tf.Variable(
            np.zeros( ((self.d_ + 1) * (self.d_ + 1), ), dtype=np.short))
        key = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.short))

        # Compute the canonical simplex
        for i in range(self.d_ + 1):
            for j in range(self.d_ - i + 1):
                canonical[i * (self.d_ + 1) + j].assign(i)
            for j in range(self.d_ - i + 1, self.d_ + 1):
                canonical[i * (self.d_ + 1) + j].assign(i - (self.d_ + 1))
        # ->> Numpify
        # canonical = tf.reshape(canonical, [self.d_ + 1, self.d_ + 1])
        # for i in range(self.d_ + 1):
        #     canonical[i, :self.d_ + 1 - i].assign(i)
        #     canonical[i, self.d_ + 1 - i:].assign(i - (self.d_ + 1))
        # canonical = tf.reshape(canonical, -1)

        # Expected standard deviation of our filter (p.6 in [Adams et al. 2010])
        inv_std_dev = np.sqrt(2. / 3.) * (self.d_ + 1)
        # Compute the diagonal part of E (p.5 in [Adams et al 2021])
        scale_factor.assign(1. / tf.sqrt( tf.cast((tf.range(self.d_) + 2) * (tf.range(self.d_) + 1), dtype=tf.float32) )  * inv_std_dev)

        # Compute the simplex each feature lies in
        for k in range(self.N_):
            # Elevate the feature (y = Ep, see p.5 in [Adams et al. 2010])
            f = feature[k]

            # sm contains the sum of 1..n of our feature vector
            sm = 0
            for j in range(self.d_, 0, -1):
                cf = f[j - 1] * scale_factor[j - 1]
                elevated[j].assign(sm - j * cf)
                sm += cf
            elevated[0].assign(sm)
            # ->> Numpify
            # cf = f * scale_factor # (d_, )
            # E = np.vstack([
            #     np.ones((self.d_, ), dtype=np.float32), 
            #     np.diag(-np.arange(self.d_, dtype=np.float32) - 2) + np.triu(np.ones((self.d_, self.d_), dtype=np.float32))])
            # elevated[:] = E.dot(cf)

            # Find the closest 0-colored simplex through rounding
            
            # for i in range(self.d_ + 1):
            #     v = down_factor * elevated[i]
            #     up = np.ceil(v) * up_factor
            #     down = np.floor(v) * up_factor
            #     if (up - elevated[i] < elevated[i] - down):
            #         rd2 = np.short(up)
            #     else:
            #         rd2 = np.short(down)

            #     rem0[i] = rd2
            #     _sum += rd2 * down_factor
            # ->> Numpify
            down_factor = 1. / (self.d_ + 1)
            up_factor = self.d_ + 1
            v = down_factor * elevated
            up = tf.math.ceil(v) * up_factor
            down = tf.math.floor(v) * up_factor
            rem0.assign(tf.where(up - elevated < elevated - down, up, down))
            _sum = tf.cast(tf.reduce_sum(rem0) * down_factor, tf.int32)

            # Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
            rank.assign(tf.zeros_like(rank))
            for i in range(self.d_):
                di = elevated[i] - rem0[i]
                for j in range(i + 1, self.d_ + 1):
                    if di < elevated[j] - rem0[j]:
                        rank[i].assign(rank[i] + 1)
                    else:
                        rank[j].assign(rank[j] + 1)

            # If the point doesn't lie on the plane (sum != 0) bring it back
            for i in range(self.d_ + 1):
                rank[i].assign(rank[i] + _sum)
                if rank[i] < 0:
                    rank[i].assign(rank[i] + self.d_ + 1)
                    rem0[i].assign(rem0[i] + self.d_ + 1)
                elif rank[i] > self.d_:
                    rank[i].assign(rank[i] - self.d_ + 1)
                    rem0[i].assign(rem0[i] - self.d_ + 1)

            # Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
            barycentric.assign(tf.zeros_like(barycentric))
            for i in range(self.d_ + 1):
                v = (elevated[i] - rem0[i]) * down_factor
                barycentric[self.d_ - rank[i]].assign(barycentric[self.d_ - rank[i]] + v)
                barycentric[self.d_ - rank[i] + 1].assign(barycentric[self.d_ - rank[i] + 1] - v)
            # Wrap around
            barycentric[0].assign(barycentric[0] + 1. + barycentric[self.d_ + 1])

            # Compute all vertices and their offset
            # ->> Numpify
            for remainder in range(self.d_ + 1):
                for i in range(self.d_):
                    key[i].assign(tf.cast(rem0[i], dtype=tf.int16) + canonical[remainder * (self.d_ + 1) + rank[i]])
                self.offset_[k, remainder].assign(self.hash_table.find(key, True))
                self.rank_[k, remainder].assign(rank[remainder])
                self.barycentric_[k, remainder].assign(barycentric[remainder])

        del scale_factor, elevated, rem0, barycentric, rank, canonical, key

        # Find the neighbors of each lattice point

        # Get the number of vertices in the lattice
        self.M_ = self.hash_table.size()

        # Create the neighborhood structure
        self.blur_neighbors_ = tf.Variable(
            np.zeros( (self.d_ + 1, self.M_, 2), dtype=np.int32))
        
        n1 = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.short))
        n2 = tf.Variable(
            np.zeros((self.d_ + 1, ), dtype=np.short))

        # For each of d+1 axes,
        for j in range(self.d_ + 1):
            for i in range(self.M_):
                key = self.hash_table.get_key(i)
                for k in range(self.d_):
                    n1[k].assign(key[k] - 1)
                    n2[k].assign(key[k] + 1)

                n1[j].assign(key[j] + self.d_)
                n2[j].assign(key[j] - self.d_)

                self.blur_neighbors_[j, i, 0].assign(self.hash_table.find(n1))
                self.blur_neighbors_[j, i, 1].assign(self.hash_table.find(n2))
        
        del n1, n2

    def seq_compute(self, out, inp, value_size, reverse):
        '''
        Compute sequentially

        Args:
            inp: (size, value_size)
            value_size: value size
            reverse: indicating the blurring order
        
        Returns:
            out: (size, value_size)
        '''
        # Shift all values by 1 such that -1 -> 0 (used for blurring)
        # values = np.zeros( ((self.M_ + 2) * value_size, ), dtype=np.float32)
        # new_values = np.zeros( ((self.M_ + 2) * value_size, ), dtype=np.float32)
        # ->> Numpify
        values = tf.Variable(
            np.zeros( ((self.M_ + 2), value_size), dtype=np.float32 ))
        new_values = tf.Variable(
            np.zeros( ((self.M_ + 2), value_size), dtype=np.float32 ))

        # Splatting
        for i in range(self.N_):
            for j in range(self.d_ + 1):
                o = self.offset_[i, j] + 1
                w = self.barycentric_[i, j]
                values[o].assign_add(w * inp[i])
        # ->> Numpify
        # np.unique?

        j_range = tf.range(self.d_, -1, -1) if reverse else tf.range(self.d_ + 1)
        for j in j_range:
            # for i in range(self.M_):
            #     old_val = values[i + 1]
            #     new_val = new_values[i + 1]

            #     n1 = self.blur_neighbors_[j, i, 0] + 1
            #     n2 = self.blur_neighbors_[j, i, 1] + 1
            #     n1_val = values[n1]
            #     n2_val = values[n2]

            #     new_val = old_val + .5 * (n1_val + n2_val)
            # ->> Numpify
            # new_vals = new_values[1:self.M_ + 1]
            n1s = self.blur_neighbors_[j, :self.M_, 0] + 1
            n2s = self.blur_neighbors_[j, :self.M_, 1] + 1
            n1_vals = tf.gather(values, n1s)
            n2_vals = tf.gather(values, n2s)

            new_values[1:self.M_ + 1].assign(values[1:self.M_ + 1] + .5 * (n1_vals + n2_vals))

            values, new_values = new_values, values

        # Alpha is a magic scaling constant (write Andrew if you reall wanna understand this)
        alpha = 1. / (1 + np.power(2., -self.d_))
        
        # Slicing
        out *= 0
        for i in range(self.N_):
            for j in range(self.d_ + 1):
                o = self.offset_[i, j] + 1
                w = self.barycentric_[i, j]
                out[i].assign_add(w * values[o] * alpha)

        del values, new_values

    def compute(self, inp, reverse=False):
        size, ch = inp.shape
        out = tf.Variable(np.zeros_like(inp))
        self.seq_compute(out, inp, ch, reverse)
        return out