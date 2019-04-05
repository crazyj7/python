import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


A=np.array([[0,0,1],[2,2,1],[1,4,1],[4,6,1]])
b=np.array([[1],[6],[2],[10]])

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


# model : y = a1 x1 + a2 x2 + a3
mx1 = np.average(A[:,0])
mx2 = np.average(A[:,1])
my = np.average(b[:,0])

# predict
print('mx1=', mx1, 'mx2=', mx2, ' my=', my)
cov=np.average((A[:,0]-mx1)*(A[:,1]-mx2)*(b[:,0]-my))
va1=np.average(np.power((A[:,0]-mx1), 2))
va2=np.average(np.power((A[:,1]-mx2), 2))

a1=cov/va1
a2=cov/va2
print('a1=',a1, 'a2=',a2)
#y = a1 x1 + a2 x2 + a3
#a3 = a1 x1^ + a2 x2^ - y^
a3 = my - a1 * mx1 - a2 * mx2
print('a3=', a3)

# predict plane
x1=np.arange(0, 7, 0.2)
x2=np.arange(0, 7, 0.2)
x1,x2 = np.meshgrid(x1,x2)
Y = a1*x1 + a2*x2+ a3
ax.plot_surface(x1, x2, Y, alpha=0.2)

ax.legend()

plt.show()

