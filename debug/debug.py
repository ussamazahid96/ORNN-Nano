import numpy as np
import matplotlib.pyplot as plt

# import torch as t 
# m = t.nn.Tanh()
# x = t.linspace(-3, 3, 256)
# y = m(x)
# x = x.numpy().reshape(-1)
# y = y.numpy().reshape(-1)
# plt.plot(x, y, '.k',)
# plt.show()
# y = np.round(y*2**7)/2**7
# np.savetxt("tanh.txt", y, fmt='%.8f')
# exit(0)

pt = np.loadtxt("input_pt.txt")
cpp = np.loadtxt("input_cpp.txt")
pycy = np.loadtxt("input_pycy.txt")

plt.plot(pt, pycy, '.k',)
plt.show()
diff = pycy-pt
print(min(diff), max(diff))
assert np.allclose(pycy, pt)