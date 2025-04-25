import numpy as np


def Sample():
    x = 0
    for _ in range(2):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x


trial = 1000
V, n = 0, 0


for i in range(trial):
    s = Sample()
    n += 1
    V += (s - V) / n
    if (i + 1) % 100 == 0:
        print(f"Trial {i + 1}: Sample mean = {V}")
