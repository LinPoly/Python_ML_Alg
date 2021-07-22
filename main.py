# %%
import numpy as np

# %%
foo_array = np.arange(18).reshape([3, 3, 2])
ano_array = np.arange(9).reshape([3, 3])

# %%
try:
    rst = foo_array*ano_array
except ValueError:
    print('Fail to broadcast.')

# %%
ano_array = np.reshape(ano_array, [3, 3, 1])
try:
    rst = foo_array*ano_array
    print(rst)
except ValueError:
    print('Fail to broadcast.')

#%%
ano_array = np.reshape(ano_array, [3, 3])
A = np.einsum('ii->i', ano_array)
B = np.einsum('ij->ji', ano_array)
C = np.einsum('ij->i', ano_array)

# %%
x = np.arange(0, 512)
y = np.arange(0, 512)
# Use meshgrid() to generate coordinate arrays, it is
# an efficient approach to produce arrays whose elements
# are computed based on their coordinates.
xx, yy = np.meshgrid(x, y)
xx = xx - 256
yy = yy - 256
H = np.sinc(np.pi*(0.1*xx + 0.1*yy))

# %%
import numpy as np
import numba as nb
import math


@nb.vectorize('float64(int64, int64)', target='parallel')
def myfunc(a, b):
    return math.sinh(a)/math.exp(b)


a = np.arange(15).reshape([5, 3])
b = np.arange(3) # ufuncs defined by numba also support broadcasting mechanism
c = np.arange(1, 16).reshape([5, 3])
