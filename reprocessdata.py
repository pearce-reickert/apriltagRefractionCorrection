
from dt_apriltags import Detector
import os
import yaml
import pickle
import cv2
import numpy as np
from datetime import datetime
import AprilTag_Detections as atd





experimentid = "static_plexiglass"
if(experimentid == "static"):
    fldr_dir = 'experiments/~Tank3/'
    reprocessed_dir = 'experiments/~Tank3_reprocessed/'
    OLDMODE = True
if(experimentid == "static_plexiglass"):
    fldr_dir = 'experiments/~Tank3/'
    reprocessed_dir = 'experiments/~Tank3_reprocessed_plexiglass/'
    OLDMODE = True
    
img_end = "Image.png"
if(OLDMODE):
    img_end = "_0Image.png"
pkl_end = "Data.pkl"

at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


files =  os.listdir(fldr_dir)
sorted_files = sorted(files)
sorted_pkls = [s for s in sorted_files if s.endswith('.pkl')]

with open('AT_info.yaml', 'r') as stream:
    parameters = yaml.load(stream)

FILEN = 0

for p in sorted_pkls:
    
    pkl_dir = fldr_dir+p
    img_dir = pkl_dir[:-len(pkl_end)]+img_end
    
    with open(pkl_dir,'rb') as inp:
        ATs_source = pickle.load(inp)
    inp.close()
    
    ATs = atd.Detections()
    #change index of refraction to reflect plexiglass instead of air
    ATs.custom_index(1,1.333)
    if(experimentid == "static_plexiglass"):
        ATs.custom_index(1.5,1.333)
    
    image = cv2.imread(img_dir)
        
    grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    

    cameraMatrix = np.array(parameters['tank_experiment_1']['K']).reshape((3,3))
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    #print("Detecting AprilTags...")
    img = grayscale_image #cv2.imread(ab_path, cv2.IMREAD_GRAYSCALE)

    tags = at_detector.detect(img, True, camera_params, parameters['tank_experiment_1']['tag_size'])

    #tag_ids = [tag.tag_id for tag in tags]
    #print(len(tags), " tags found: ", tag_ids)


    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    #print(len(color_img))
    cv2.circle(color_img,(320,240),1,(0,255,255),-1)           
    for tag in tags:
        print()
        print(ATs.air_index)
        print(ATs.water_index)
        print(ATs.n)
        print()
        ATs.up(tag)
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
        #xyz = [round(num*100, 2) for num in ATs.locs[tag.tag_id]]
        #print(xyz)
        annotation = str(tag.tag_id)#str(xyz)#
        cv2.putText(color_img, annotation,
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))
    
        
    filestr = reprocessed_dir+f'{FILEN:05d}'
    
    #print(filestr)
    
    with open(filestr+'Data.pkl','wb') as outp:
        pickle.dump(ATs,outp,-1)
    cv2.imwrite(filestr+'Image.png',color_img)
    
    FILEN = FILEN+1
    
    ATs.reset_update()
    
    cv2.imshow('Apriltag Detections',  color_img)
    key = cv2.waitKey(1)
