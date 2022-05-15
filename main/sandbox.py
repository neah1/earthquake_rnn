import numpy as np

temp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
T = 5
H = 6
start = -(T + H)
end = -H if H > 0 else None
res = temp[start:end]

print(res)
