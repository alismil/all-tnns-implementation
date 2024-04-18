import math

in_dim = 48
k_dim = 48
out_dim = 50

for c in range(1, 3000):
    for p in range(in_dim//2):
        for s in range(1, in_dim - k_dim + 2):
            if math.sqrt(c) * (math.floor((in_dim - k_dim + 2 * p) / s) + 1) == out_dim:
                print(c, p, s)
