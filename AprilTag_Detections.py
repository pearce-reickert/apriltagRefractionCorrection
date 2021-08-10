import numpy as np
from dt_apriltags import Detection
from datetime import datetime
import math
import aprilTagLocations as atl
import matplotlib.pyplot as plt

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

def PlotApriltagHomography(ax,H):
    #H is a 4x4 homography matrix
    #a-d are the 4 tag corners, e is 1 in the z direction, o is the origin
    s = 0.074
    D = np.array([[0,0,0,1],[0,0,s,1],[s,s,0,1],[-s,s,0,1],[-s,-s,0,1],[s,-s,0,1]]).T
    HD = np.matmul(H,D)
    #print(H)
    #print(D)
    #print(HD)
    #print()
    
    return HD[0,:],HD[1,:],HD[2,:]

class Detections():
    def __init__(self):
        self.timestamp = datetime.now()
        self.Tags = []
        self.Tags = [Detection() for i in range(50)]
        self.blank = Detection()
        self.locs = [np.array([0,0,0]) for tag in self.Tags]
        self.corrected_locs = [np.array([0,0,0]) for tag in self.Tags]
        self.corrected_matrix = [np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) for tag in self.Tags]
        self.updated = [False for tag in self.Tags]
        
        self.water_point = np.array([0,0,0.01])
        self.water_normal = np.array([0,0,-1])
        
        self.acrylic_index = 1.491
        self.air_index = 1
        self.water_index = 1.333
        self.n = self.air_index/self.water_index
        
        self.e_printed = False
        
        self.fp = ""
        self.prev_fp = ""
        self.fp_idx = 0
        
        #self.written = [False for i in range(50)]
        tag_ids = [tag.tag_id for tag in self.Tags]
        print(len(self.Tags), " tags found: ", tag_ids)
    
    def snell(self,s1):
        N = self.water_normal/np.linalg.norm(self.water_normal)
        s1 = s1/np.linalg.norm(s1)
        n = self.n
        #print(type(n))
        Nxs1 = np.cross(N,s1)
        t = np.dot(Nxs1,Nxs1)
        #print(n**2)
        D = 1-n**2*t
        B = np.sqrt(D)
        A = n*np.cross(N,np.cross(-N,s1))
        s2 = A-N*B
        return s2
    
    def calc_theta(self,s):
        Na = abs(self.water_normal)
        sa = abs(s)
        Nl = np.linalg.norm(self.water_normal)
        sl = np.linalg.norm(s)
        A = np.dot(sa,Na)
        #print(A)
        B = (Nl*sl)
        #print(B)
        theta = math.acos(A/B)
        #print(theta)
        return theta
    
    def reset_update(self):
        self.updated = [False for tag in self.Tags]
    
    def up(self, AT):
        #print(AT.tag_id)
        #self.written[AT.tag_id]= True
        self.Tags[AT.tag_id]= AT
        self.updated[AT.tag_id] = True
        
        d = np.array([0,0,0,1])
        try:
             p_R = np.array(AT.pose_R)
             p_t = np.array(AT.pose_t)
             p_Rt = np.concatenate([p_R,p_t],axis=1)
             #print(p_R.shape,p_t.shape,p_Rt.shape)
             #print(np.concatenate((p_R,p_t),axis=1))
             #print(np.matmul(p_Rt,d))
        except ValueError as error:
            pass
            #print("Val:",error)
        except AttributeError as error:
            pass
            #print("Att:",error)
        
        L = np.matmul(p_Rt,d)
        self.locs[AT.tag_id]=L
        L_c = self.correct(L)
        self.corrected_locs[AT.tag_id]=L_c
        p = np.matrix(L_c)
        tag_in_cam_frame = atl.RpToTf(p_R,p)
        cam_in_tag_frame = atl.RpToInvTf(p_R,p)
        
        tag_in_world_frame = atl.idMap[AT.tag_id]
        
        cam_in_world_frame = np.matmul(tag_in_world_frame,np.linalg.inv(tag_in_cam_frame))
        cam_in_world_frame2 = np.matmul(tag_in_world_frame,cam_in_tag_frame)
        
        #print("euler angles:",atl.rot2eul(p_R))
        #print(tag_in_cam_frame)
        #print(cam_in_tag_frame)
        #print(atl.tfToVec(tag_in_world_frame))
        #print(atl.tfToVec(cam_in_world_frame))
        #print(atl.tfToVec(cam_in_world_frame2))
        #print(cam_in_world_frame)
        print(cam_in_world_frame2)
                
        self.corrected_matrix[AT.tag_id] = cam_in_world_frame2
    
    def ticf_raw(self,num):
        p_R = np.array(self.Tags[num].pose_R)
        p = np.matrix(self.locs[num]).T
        #print(p_R)
        #print(p)
        #print(atl.RpToTf(p_R,p))
        #print()
        return atl.RpToTf(p_R,p)
    
    def ticf_corrected(self,num):
        p_R = np.array(self.Tags[num].pose_R)
        p = np.matrix(self.corrected_locs[num]).T
        #print(p_R)
        #print(p)
        #print(atl.RpToTf(p_R,p))
        #print()
        return atl.RpToTf(p_R,p)
    
    def correct(self,wrong_loc):
        #takes wrong_loc, a 3 element vector and corrects for refraction due to the water's surface
        #1. Finds intersection with the predefined water's surface
        intersection_point = LinePlaneCollision(self.water_normal, self.water_point, wrong_loc, [0,0,0])
        
        d_illusory = np.linalg.norm(wrong_loc-intersection_point)
        
        true_dir = self.snell(wrong_loc)
        
        theta_i = self.calc_theta(wrong_loc)
        
        theta_r = self.calc_theta(true_dir)
        
        d_true = math.cos(theta_r)**2/math.cos(theta_i)**2/self.n*d_illusory
        
        true_loc = intersection_point+d_true*true_dir
        
        
#         print('------------------------')
#         print('wrong loc:         ',wrong_loc)
#         print('intersection point:',intersection_point)
#         print('illusory distance: ',d_illusory)
#         print('true_direction:    ',true_dir)
#         print('theta_i:           ',theta_i)
#         print('theta_r:           ',theta_r)
#         print('theta_r check:     ',math.asin(self.n*math.sin(theta_i)))
#         print('true distance:     ',d_true)
#         print('assumed depth:     ',d_illusory/self.n)
#         print('true loc:          ',true_loc)
        
        return true_loc
           
    
    def cam_from_3d(self,P):
        
        w = 1600
        h = 896
        
        c_w = 1.25
        c_h = 0.71#13.5/23.5
        #print(P)
        t = [P[0]/P[2],P[1]/P[2]]
        #print(t)
        n = [(t[0]+c_w/2)/c_w,(t[1]+c_h/2)/c_h]
        #print(n)
        p = [int(n[0]*w),int(n[1]*h)]
        #print(p)
        return p
    
    def highlight_tag_distance(self,a,b,color_img,correct):
        f = 100
        
        L1 = self.locs[a]*f
        L2 = self.locs[b]*f
        
        if(correct == True):
            L1 = self.corrected_locs[a]*f
            L2 = self.corrected_locs[b]*f
        
        p1 = self.cam_from_3d(L1)
        p2 = self.cam_from_3d(L2)
        
        
        pointa = np.array([num for num in L1])
        pointb = np.array([num for num in L2])
        
        distance = np.linalg.norm(pointa-pointb)
        
        annotation = str(round(distance,3))
        
        cv2.putText(color_img, annotation,
                org=(int((p1[0]+p2[0])/2)-40,int((p1[1]+p2[1])/2)-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 255))
        
        cv2.line(color_img, tuple(p1), tuple(p2), (255,255,0))
            
        
        
    def print(self):
        tag_ids = [tag.tag_id for tag in self.Tags]
        print(len(self.Tags), " tags found: ", tag_ids)
        #print(type(self.locs[1]))
        print(np.array(self.locs))
        print(np.array(self.corrected_locs))
        #for AT in self.Tags:
            #print(AT)
