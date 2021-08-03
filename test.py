import pickle
import AprilTag_Detections as atd


with open('Apriltag_data.pkl','rb') as inp:
    AT_data = pickle.load(inp)
    AT_data.print()