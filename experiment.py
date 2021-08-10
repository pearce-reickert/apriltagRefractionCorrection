import pickle
import AprilTag_Detections as atd
import aprilTagLocations as atl
import os
import sys
import cv2
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def show_tank(fig,ax):
    yb = np.array([-1.42,1.52])
    xb = np.array([-2.31,2.15])
    zb = np.array([0,1.07])

    P0 = np.array([xb[0],yb[0],zb[0]])
    P1 = np.array([xb[0],yb[0],zb[1]])
    P2 = np.array([xb[0],yb[1],zb[1]])
    P3 = np.array([xb[0],yb[1],zb[0]])
    P4 = np.array([xb[1],yb[1],zb[0]])
    P5 = np.array([xb[1],yb[1],zb[1]])
    P6 = np.array([xb[1],yb[0],zb[1]])
    P7 = np.array([xb[1],yb[0],zb[0]])

    Z = np.array([P0,P1,P2,P3,P7,P6,P5,P4])

    #print(Z)

    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    verts = [[Z[0],Z[1],Z[2],Z[3]],
     [Z[4],Z[5],Z[6],Z[7]], 
     [Z[0],Z[1],Z[5],Z[4]], 
     [Z[2],Z[3],Z[7],Z[6]], 
     [Z[1],Z[2],Z[6],Z[5]],
     [Z[4],Z[7],Z[3],Z[0]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
     facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    
    


fldr_dir = 'experiments/~Tank3/'

img_end = "_0Image.png"
pkl_end = "Data.pkl"

files =  os.listdir(fldr_dir)
sorted_files = sorted(files)
sorted_pkls = [s for s in sorted_files if s.endswith('.pkl')]
#print(sorted_pkls)

fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')


cam_H = np.eye(4)
    
GT_tag_ID = 40
GT_tag_in_world_frame = atl.idMap[GT_tag_ID]
for p in sorted_pkls:
    
    pkl_dir = fldr_dir+p
    img_dir = pkl_dir[:-len(pkl_end)]+img_end
    #print(pkl_dir)
    #print(img_dir)
    
    with open(pkl_dir,'rb') as inp:
        AT_data = pickle.load(inp)
        #AT_data.print()
        
        GT_tag_in_camera_frame = AT_data.ticf_corrected(GT_tag_ID)
        GT_cam_in_world_frame = np.matmul(GT_tag_in_world_frame,np.linalg.inv(GT_tag_in_camera_frame))


        
        for i in range(0,42):
            #print(i)
            if(AT_data.updated[i]):
                color = 'blue'
                if(i == GT_tag_ID):
                    color = 'green'
                H = np.matmul(GT_cam_in_world_frame,AT_data.ticf_corrected(i))
                #H = AT_data.ticf_raw(i)
                [x,y,z] = atd.PlotApriltagHomography(ax,H)
                ax.scatter3D(x,y,z,c=color)
                
                color = 'red'
                H = np.matmul(GT_cam_in_world_frame,AT_data.ticf_raw(i))
                #H = AT_data.ticf_raw(i)
                [x,y,z] = atd.PlotApriltagHomography(ax,H)
                ax.scatter3D(x,y,z,c=color)
            
                GT_H = atl.idMap[i]
                [x,y,z] = atd.PlotApriltagHomography(ax,GT_H)
                ax.scatter3D(x,y,z,c='black')

    [x,y,z] = atd.PlotApriltagHomography(ax,GT_cam_in_world_frame)
    ax.scatter3D(x,y,z,c='cyan')
    ax.view_init(180, 90) 
    ax.set_xlim(-3,0)
    ax.set_ylim(-1,2)
    ax.set_zlim(0,3)
    show_tank(fig,ax)
    
    plt.pause(0.05)  
    
    image = cv2.imread(img_dir)
    
    cv2.imshow('image',image)
    cv2.waitKey(0)
    
    plt.cla()


plt.show()
cv2.destroyAllWindows()


