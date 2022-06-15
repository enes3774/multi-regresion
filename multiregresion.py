import numpy as np

n=2
data_num=8
matrix=np.zeros((n+1,n+2))
data_x=[[1,1,2],[1,2,3],[1,2,1],[1,4,2],[1,5,2],[1,6,4],[1,9,5],[1,9,1]]
def f(x1,x2):
    return 3*x1+5*x2+8
data_y=[]
for x in data_x:
    data_y.append(f(x[1],x[2]))
data_x=np.array(data_x)
data_y=np.array(data_y).reshape(8,1)
data=np.concatenate((data_x,data_y),axis=1)


for i in range(n+1):
    for j in range(n+2):
        for e in range(data_num):
            
            matrix[i][j]+=data[e][i]*data[e][j]
##############################################gauss############################################

for i in range(n):
    oran=matrix[i+1][0]/matrix[0][0]
    for j in range(n+2):
        matrix[i+1][j]-=oran*matrix[0][j]
oran=matrix[2][1]/matrix[1][1]
for i in range(n+1):
    matrix[2][i+1]-=oran*matrix[1][i+1]
###################################### w leri ve b yi bul####################################

w2=matrix[2][3]/matrix[2][2]
w1=(matrix[1][3]-w2*matrix[1][2])/matrix[1][1]
b=(matrix[0][3]-w2*matrix[0][2]-w1*matrix[0][1])/matrix[0][0]

###################################### plot ####################################


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def f_pred(x1, x2):
    return x1*w1+x2*w2+b
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
x1 = np.linspace(-10,10,50)
x2 = np.linspace(-10,10,50)
X1, X2 = np.meshgrid(x1, x2)
Z=f_pred(X1,X2)
ig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, X2, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

for i in range(len(data)):
    ax.scatter(data[i,1],data[i,2],data[i,3])
    print(data[i,1],data[i,2],data[i,3])

    plt.show()
