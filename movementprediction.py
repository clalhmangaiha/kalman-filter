import matplotlib.pyplot as plt
import numpy as np

x0=np.linspace(10,80,8)
y0 = [0.6,0.68,0.72,0.76,0.8,0.75,0.68,0.6]
plt.xlabel("Grid Size")
plt.ylabel("Movement Prediction accuracy")
plt.ylim(0,1)
plt.plot(x0,y0,'g-',x0,y0,'ro')
plt.show()
