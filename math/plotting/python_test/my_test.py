#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


players = np.array(['Wembanyama','Sochan', 'Bassey','Vassel', 'Colins'])
heights = np.array([2.13, 2.03, 2.08, 1.96, 2.11])
points = np.array([19.0, 11.0, 3.3 ,18.1 ,13.7])
colors = np.array(['r', 'b', 'g', 'k','y'])
plt.bar(players, heights, np.divide(points,10), color=colors, align='edge')

plt.title('Height and points Baskeball')
plt.xlabel('Joueurs')
plt.ylabel('taille des joueurs')
plt.legend()
plt.show()


"""np array creation"""
"""
arr = np.array([[1,2,3],[4,7,6]])
new_arr = arr[:,:]
arr2 = np.arange(0,11)


x = np.array([[1,2],[3,4]])
y = np.array([[5,7],[6,9]])

x1 = x[:,0]
y1 = y[:,0]

x2 = x[:,1]
y2 = y[:,1]


ax = np.random.rand(50)*10
ay = np.random.rand(50)*10
print('ax')
print(ax)

"""

"""plt.plot(x,y)"""
"""
plt.plot(x1,y1,"b",label="line 1")
plt.plot(x2,y2,"g",label="line 2")
plt.scatter(ax,ay)

plt.text(1.5,5.5,'line1')
plt.text(2.5,8.5,'line2')
"""

"""obj = {'xlabel':x, 'ylabel':y}"""
"""plt.plot('xlabel','ylabel',data=obj)
"""

"""
print("x1:")
print(x1)
print("y1:")
print(y1)

print("x2:")
print(x2)
print("y2:")
print(y2)


print("x:")
print(x)
print("y:")
print(y)
"""

""" categorical datas """
