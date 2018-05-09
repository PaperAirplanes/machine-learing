import numpy as np

a = np.array([[1,20],[2,18],[3,19]])
b = a.min(0)
c = a.max(0)
print(b)
print(c)