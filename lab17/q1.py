#Implement a feature mapping function called Transform()
#Use following set of samples
#Plot these points.  Then transform these points using your “Transform” function into 3-dim space.
# Plot the points and manipulate the points so that you can see a separating plane in 3D.


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


red_points=np.array([[3,15],[6,6],[6,11],[9,5],[10,10],[11,5],[12,6],[16,3]])
blue_points=np.array([[1,13],[1,18],[2,9],[3,6],[6,3],[9,2],[13,1],[18,1]])

#------------------------------------------
#--------Transform function----------------
#------------------------------------------
#so transform function adds new feature which is achieved by Z=X1.X2
#Z is the third co-ordinate and also separating plane
def transform(points):
  x1=points[:,0]
  x2=points[:,1]
  Z=x1*x2
  return np.column_stack((x1,x2,Z))
#transform the points
red_3d=transform(red_points)
blue_3d=transform(blue_points)
print('red_points after transformation:',red_3d)
print('blue_points after transformation:',blue_3d)

#-------------------------------------------------
#--------------------2D--------------------------
#-------------------------------------------------
#plot the function
plt.figure(figsize=(8,8))
plt.scatter(red_points[:,0],red_points[:,1],label='Red',s=80)
plt.scatter(blue_points[:,0],red_points[:,1],label='Blue',s=80,marker='^')

for x,y in red_points:
    plt.text(x+0.2,y+0.2,f'{x},{y}')
for x,y in blue_points:
    plt.text(x+0.2,y+0.2,f'{x},{y}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2-Dimensional Plot')
plt.legend()
plt.grid(True)
plt.show()

#----------------------------------------------------
#--------------3D------------------------------------
#----------------------------------------------------

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111, projection='3d')#111:1-row,1-col,1-plot
ax.scatter(red_3d[:,0],red_3d[:,1],red_3d[:,2],label='Red',s=80)
ax.scatter(blue_3d[:,0],blue_3d[:,1],blue_3d[:,2],label='Blue',s=80,marker='^')

#separating plane:z=27
x_range=np.linspace(0,20,10)
y_range=np.linspace(0,20,10)
x_plane,y_plane=np.meshgrid(x_range,y_range)
z_plane=np.full_like(x_plane,27)
ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.3)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x1*x2")
ax.set_title("3D Transformed Data with Separating Plane z = 27")
ax.legend()

# Good viewing angle to see the separation
ax.view_init(elev=25, azim=45)

plt.show()
