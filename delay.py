import matplotlib.pyplot as plt
import numpy as np

ox = np.linspace(1,50,10)
oy = [10,9.9,9.8,9.7,9.6,9.7,9.8,9.7,9.6,9.7]

plt.plot(ox,oy,'g',ox,oy,'ro')
plt.ylabel("Hand-off Delay")
plt.xlabel("Movement Prediction Accuracy")
plt.ylim(0,15)
# plt.yscale(10)
plt.grid(b=None, which='major', axis='both')
plt.xlim(0)

plt.show()
