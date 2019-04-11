import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


A=np.array([[0,0,1],[2,2,1],[1,4,1],[4,6,1]])
# b=np.array([[1],[6],[2],[10]])
b=np.array([[0],[7],[3],[10]])

x = np.dot( np.dot( np.linalg.inv( np.dot(np.transpose(A), A)), np.transpose(A) ) , b)

print('A=',A)
print('b=',b)
print('x=',x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(A[:,0], A[:,1], b[:,0], marker='o', label='sample')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Y')
y = x[0]*A[:,0] + x[1]*A[:,1]+ x[2]
print('y=', y)
ax.scatter(A[:,0], A[:,1], y, marker='+', label='fitting')

# predict plane
x1=np.arange(0, 7, 0.2)
x2=np.arange(0, 7, 0.2)
x1,x2 = np.meshgrid(x1,x2)
Y = x[0]*x1 + x[1]*x2+ x[2]
ax.plot_surface(x1, x2, Y, alpha=0.2)


ax.legend()

plt.show()

