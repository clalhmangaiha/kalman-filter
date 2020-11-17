from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import CirclePolygon

someX, someY = 0.5, 0.5
plt.figure()
currentAxis = plt.gca()





#GRID 1 
currentAxis.add_patch(Rectangle((0.0  , 0.5 ), 0.5, 0.5, fill=None, alpha=1))
currentAxis.add_patch(Rectangle((0.1  , 0.6 ), 0.3, 0.3, color='Blue',alpha=0.2))
# currentAxis.add_patch(Circle((0.22,0.7), radius=0.04))
l = plt.plot(0.25,0.74, 'ro',markersize=30,markerfacecolor='C0',alpha=0.6)
plt.text(0.24,0.73,"A")


#GRID 2
currentAxis.add_patch(Rectangle((someX  , someY ), 0.5, 0.5, fill=None, alpha=1))
currentAxis.add_patch(Rectangle((0.60 , 0.60 ), 0.3, 0.3, color='Blue',alpha=0.2))
l = plt.plot(0.75,0.74, 'ro',markersize=30,markerfacecolor='C0',alpha=0.6)
plt.text(0.74,0.73,"B")

#GRID 3
currentAxis.add_patch(Rectangle((0.0  , 0.0 ), 0.5, 0.5, fill=None, alpha=1))
currentAxis.add_patch(Rectangle((0.10 , 0.10 ), 0.3, 0.3, color='Blue',alpha=0.2))
l = plt.plot(0.24,0.26, 'ro',markersize=30,markerfacecolor='C0',alpha=0.6)
plt.text(0.23,0.25,"C")

#GRID 4
currentAxis.add_patch(Rectangle((0.5  , 0.0 ), 0.5, 0.5, fill=None, alpha=1))
currentAxis.add_patch(Rectangle((0.60 , 0.10 ), 0.3, 0.3, color='Blue',alpha=0.2))
l = plt.plot(0.74,0.26, 'ro',markersize=30,markerfacecolor='C0',alpha=0.6)
plt.text(0.73,0.25,"D")




plt.show()