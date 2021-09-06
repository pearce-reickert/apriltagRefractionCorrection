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


def plot_error(agg_error):
    
    #print(agg_error)
    
    agg_error = np.array(agg_error)
    #print(agg_error)

    e_vs_dist = agg_error[:,[0,2,3]]
    e_vs_angle = agg_error[:,[1,2,3]]
    #print(e_vs_dist)


    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.scatter(e_vs_dist[:,0],e_vs_dist[:,1],c = 'r',label = 'uncorrected')
    ax1.scatter(e_vs_dist[:,0],e_vs_dist[:,2],c = 'g',label = 'corrected')
    plt.title('Tag Error vs. Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('Error (m)')
    plt.legend(loc='upper left');

    ax2 = fig.add_subplot(222)
    ax2.scatter(e_vs_angle[:,0],e_vs_angle[:,1],c = 'r')
    ax2.scatter(e_vs_angle[:,0],e_vs_angle[:,2],c = 'g')
    plt.title('Tag Error vs. Angle')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Error (m)')
    
    ucorr = e_vs_angle[:,1]
    corr = e_vs_angle[:,2]
    
    P = (1-(corr/ucorr))*100
    
    ax3 = fig.add_subplot(224)
    ax3.scatter(e_vs_angle[:,0],P,c = P)
    plt.title('%change in error from algorithm vs. Angle')
    plt.xlabel('Angle (rad)')
    plt.ylabel('Error (m)')
    plt.ylim([-100, 100])

    ax4 = fig.add_subplot(223)
    ax4.scatter(e_vs_dist[:,0],P,c = P)
    plt.title('%change in error from algorithm vs. Distance')
    plt.xlabel('Distance (m)')
    plt.ylabel('Error (m)')
    plt.ylim([-100, 100])
    
    plt.show()


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
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

FASTMODE = False
VERYFASTMODE = False
experimentid = "static_reprocessed_plexiglass"

if(experimentid == "dynamic"):
    fldr_dir = 'experiments/Tank_motion/'
    OLDMODE = False
    
elif(experimentid == "static"):
    fldr_dir = 'experiments/~Tank3/'
    OLDMODE = True
    
elif(experimentid == "static_end"):
    fldr_dir = 'experiments/~Static_end/'
    OLDMODE = True
    
elif(experimentid == "static_reprocessed"):
    fldr_dir = 'experiments/~Tank3_reprocessed/'
    OLDMODE = False    
    
elif(experimentid == "static_reprocessed_plexiglass"):
    fldr_dir = 'experiments/~Tank3_reprocessed_plexiglass/'
    OLDMODE = False   

img_end = "Image.png"
if(OLDMODE):
    img_end = "_0Image.png"
pkl_end = "Data.pkl"

files =  os.listdir(fldr_dir)
sorted_files = sorted(files)
sorted_pkls = [s for s in sorted_files if s.endswith('.pkl')]
#print(sorted_pkls)

fig = plt.figure(figsize=(4,4))
plt.pause(0.01)

ax = fig.add_subplot(111, projection='3d')


cam_H = np.eye(4)

SET_GT_TAG = False

AGG_error = []

if(VERYFASTMODE == True):
    with open(fldr_dir+'error/agg_error.pkl','rb') as inp:
        AGG_error = pickle.load(inp)
    plot_error(AGG_error)
    inp.close()
    exit
    

for p in sorted_pkls:
    
    pkl_dir = fldr_dir+p
    img_dir = pkl_dir[:-len(pkl_end)]+img_end
    #print(pkl_dir)
    #print(img_dir)
    
    with open(pkl_dir,'rb') as inp:
        AT_data = pickle.load(inp)
        #AT_data.print()
        print(AT_data.air_index)
        
        for i in range(0,42):
            if(AT_data.updated[i]):
                if(SET_GT_TAG == False):
                    GT_tag_ID = 40
                    if(experimentid == "dynamic"):
                        GT_tag_ID = i
                    GT_tag_in_world_frame = atl.idMap[GT_tag_ID]
                    GT_tag_in_camera_frame = AT_data.ticf_corrected(GT_tag_ID)
                    GT_cam_in_world_frame = np.matmul(GT_tag_in_world_frame,np.linalg.inv(GT_tag_in_camera_frame))
                    #GT_cam_in_world_frame = np.array([[-0.37400621 , 0.30864634, -0.87456091 ,  1.49],
                    # [-0.92476102 ,-0.19555039  ,0.32646149 ,-1.12],
                    # [-0.07025958 , 0.93085847  ,0.35856116 , 0.12],
                    # [ 0.         , 0.          ,0.         ,1.        ]])
                    print(GT_cam_in_world_frame)
                    SET_GT_TAG = True
                    
                if(FASTMODE == False):
                    color = 'blue'
                    #print(i,": ",SET_GT_TAG, ", ", GT_tag_ID)
                    if(i == GT_tag_ID):
                        color = 'green'
                    H = np.matmul(GT_cam_in_world_frame,AT_data.ticf_corrected(i))
                    #H = AT_data.ticf_raw(i)
                    [x,y,z] = atd.PlotApriltagHomography(H)
                    ax.scatter3D(x,y,z,c=color)
                    
                    color = 'red'
                    H = np.matmul(GT_cam_in_world_frame,AT_data.ticf_raw(i))
                    #H = AT_data.ticf_raw(i)
                    [x,y,z] = atd.PlotApriltagHomography(H)
                    ax.scatter3D(x,y,z,c=color)
                
                    GT_H = atl.idMap[i]
                    [x,y,z] = atd.PlotApriltagHomography(GT_H)
                    ax.scatter3D(x,y,z,c='black')
                
                
                
                error = AT_data.error_vector(i,GT_cam_in_world_frame)
                #print(error)
                
                if(i != GT_tag_ID):
                    AGG_error.append(error)
                #print(AGG_error)
    
    if(FASTMODE == False):
        if(SET_GT_TAG):
            [x,y,z] = atd.PlotApriltagHomography(GT_cam_in_world_frame)
            ax.scatter3D(x,y,z,c='cyan')
            
        ax.view_init(205, 100) 
        ax.set_xlim(-3,3)
        ax.set_ylim(-2,2)
        ax.set_zlim(0,3)
        show_tank(fig,ax)
        
        #plt.pause(0.0005)
        fig.canvas.draw()
        #print(img_dir)
        image = cv2.imread(img_dir)
        
        cv2.imshow('image',image)
        key = cv2.waitKey(0)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break
    
        plt.cla()
    
    SET_GT_TAG = False
    


plot_error(AGG_error)



with open(fldr_dir+'error/agg_error.pkl','wb') as outp:
    pickle.dump(AGG_error,outp,-1)
outp.close()




#plt.show()
cv2.destroyAllWindows()


