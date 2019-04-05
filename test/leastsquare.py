import numpy as np
import matplotlib.pyplot as plt

# some data
x=np.random.rand(7)*10
print('x=',x)

# black box + noise
y=x*np.random.randn()*2 + np.random.rand(len(x))*3
print('y=', y)

# model : y = ax + b
mx = np.average(x)
my = np.average(y)

print('mx=', mx, ' my=', my)
a1=np.average((x-mx)*(y-my))
a2=np.average(np.power((x-mx), 2))
a=a1/a2
b=my - a * mx
print('a=',a, 'b=',b)

plt.figure()
plt.scatter(x,y, marker='o', c='blue')

# predict
newx=np.arange(0, 10, 0.2)
yy = a*newx + b
print('predict yy=', yy)
plt.scatter(newx,yy, marker='+', c='red')

plt.show()
