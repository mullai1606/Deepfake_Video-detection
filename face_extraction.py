#Face extraction and storage
import os
import shutil
from setAspectRatio import setaspectratio
import cv2
import face_detection

def face_extract(storagepath,path):
    detector = face_detection.FaceDetector()
    #dim=(w,h) 
    dim=(299,299)
    b=1
#storagepath="/content/drive/MyDrive/sample-faces/"+vidname   #path to store extracted faces
    if os.path.exists(storagepath):
       shutil.rmtree(storagepath, ignore_errors=True); #if path already exists delete it
    os.makedirs(storagepath);
    os.chdir(storagepath)
    for keyframe in os.listdir(path):
       frame=setaspectratio(path+"/"+keyframe)
       rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       #frame=cv2.imread(path+"/"+keyframe)
       #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       boxes = detector.detect(rgbframe)
       # lets's draw boxes, just multiply each predicted [0, 1] relative coordinate to image side in pixels respectively
       for box in boxes:
          lx = int(round(box[0] * frame.shape[1]))
          ly = int(round(box[1] * frame.shape[0]))
          rx = int(round(box[2] * frame.shape[1]))
          ry = int(round(box[3] * frame.shape[0]))
          # x, y, w, h here
          #ax.add_patch(Rectangle((lx,ly),rx - lx,ry - ly,linewidth=2,edgecolor='r',facecolor='none'))
          frame2 = frame[ly:ry, lx:rx]
          rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
          face = cv2.resize(rgb, dim, interpolation=cv2.INTER_LANCZOS4)
          cv2.imwrite("face"+str(b)+".jpg", face)#Detected face
          b=b+1