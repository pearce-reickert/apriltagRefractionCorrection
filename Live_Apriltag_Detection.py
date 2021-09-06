import io
import time
import threading
#from PIL import Image
#from datetime import datetime
#from mpl_toolkits import mplot3d
#import matplotlib.pyplot as plt
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import picamera
from datetime import datetime
from dt_apriltags import Detector
#from dt_apriltags import Detection
#import math
#import aprilTagLocations as atl
import AprilTag_Detections as atd

import pickle

try:
    import cv2
except:
    raise Exception('You need cv2 in order to run the demo. However, you can still use the library without it.')
# 
try:
    from cv2 import imshow
except:
    print("The function imshow was not implemented in this installation. Rebuild OpenCV from source to use it")
    print("VIsualization will be disabled.")
    visualization = False

try:
    import yaml
except:
    raise Exception('You need yaml in order to run the tests. However, you can still use the library without it.')



# Create a pool of image processors
done = False
lock = threading.Lock()
pool = []

global prev_fp


filepath='/home/pi/Desktop/apriltagRefractionCorrection'

with open(filepath + '/AT_info.yaml', 'r') as stream:
    parameters = yaml.load(stream)


print("\n\nLive Apriltag Detection")
cv2.namedWindow('Apriltag Detections', cv2.WINDOW_AUTOSIZE)

        
global ATs
ATs = atd.Detections()


def extract_Apriltags(image, at_detector, visualization):
    grayscale_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    cameraMatrix = np.array(parameters['tank_experiment_1']['K']).reshape((3,3))
    camera_params = ( cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2] )

    #print("Detecting AprilTags...")
    img = grayscale_image #cv2.imread(ab_path, cv2.IMREAD_GRAYSCALE)

    tags = at_detector.detect(img, True, camera_params, parameters['tank_experiment_1']['tag_size'])

    #tag_ids = [tag.tag_id for tag in tags]
    #print(len(tags), " tags found: ", tag_ids)


    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    
    if visualization:
        #print(len(color_img))
        cv2.circle(color_img,(320,240),1,(0,255,255),-1)           
        for tag in tags:
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
            
        #ATs.print()
        #ATs.plot()
        
        
    if(ATs.updated[0]==True and ATs.updated[1]==True):        
        pointa = np.array([round(num*100, 2) for num in ATs.locs[0]])
        pointb = np.array([round(num*100, 2) for num in ATs.locs[1]])
        
        distance = np.linalg.norm(pointa-pointb)
        
        print('apparent distance:',distance)
        
        pointa = np.array([round(num*100, 2) for num in ATs.corrected_locs[0]])
        pointb = np.array([round(num*100, 2) for num in ATs.corrected_locs[1]])
        
        distance = np.linalg.norm(pointa-pointb)
        print('true distance:',distance)
        ATs.e_printed = False
        
    elif(ATs.e_printed == False):
        print('one or both tags obscured')
        ATs.e_printed = True
    
    
    dt_string = datetime.now().strftime("%m%d%H%M%S")
    ATs.fp = 'experiments/'+dt_string
    
    if(ATs.fp == ATs.prev_fp):
        ATs.fp_idx = ATs.fp_idx+1
    else:
        ATs.fp_idx = 0
    
    ATs.prev_fp = ATs.fp  
    
    fpn = ATs.fp+'_'+str(ATs.fp_idx)

    
    with open(fpn+'Data.pkl','wb') as outp:
        pickle.dump(ATs,outp,-1)
    cv2.imwrite(fpn+'Image.png',image)
    
    
    ATs.reset_update()
    
    
    cv2.imshow('Apriltag Detections',  color_img)#+ filename#


class ImageProcessor(threading.Thread):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)
        #self.fig = plt.figure()
        #self.ax = plt.axes(projection='3d')

    def run(self):
        # This method runs in a separate thread
        global done
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    image = cv2.imdecode(np.frombuffer(self.stream.read(), np.uint8), 1)
                    #image = cv2.imread('/home/pi/Desktop/1.jpg',1)
                    #...
                    extract_Apriltags(image, self.at_detector, visualization)
                                        
                    k = cv2.waitKey(1)
                    if k == 27:         # wait for ESC key to exit
                        done=True
                        #cv2.destroyAllWindows()
                    #done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    with lock:
                        pool.append(self)

def streams():
    while not done:
        with lock:
            if pool:
                processor = pool.pop()
            else:
                processor = None
        if processor:
            yield processor.stream
            processor.event.set()
        else:
            # When the pool is starved, wait a while for it to refill
            time.sleep(0.1)




with picamera.PiCamera() as camera:
    visualization = True
    pool = [ImageProcessor() for i in range(1)]
    camera.resolution = (640, 480)
    camera.framerate = 30
    #camera.start_preview()
    #plt.show()
    time.sleep(2)
    camera.capture_sequence(streams(), use_video_port=True)

    # Shut down the processors in an orderly fashion
    while pool:
        with lock:
            processor = pool.pop()
        processor.terminated = True
        processor.join()
 
