import matplotlib.pyplot as plt
import cv2
import framesplit
import keyframe_Extraction
import face_detection
from setAspectRatio import setaspectratio
import face_extraction
import prediction
inputvidpath = input()
modelpath = input()
vidname= inputvidpath.split("/")[-1]

path="Out/sample-images/"+vidname
framepath = path+"/splitFrames/"
facestoragepath= path+"ExtractedFrames"+vidname

framesplit.split(inputvidpath, framepath)
keyframe_Extraction.keyFrameFinding(path=framepath)
keyframe_Extraction.remove(path=framepath)

detector = face_detection.FaceDetector()
# A sample face detection image
#frame=cv2.imread(path+"/key1.jpg")
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame=setaspectratio(framepath+"/key1.jpg")
rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
#ax.imshow(frame)
dim=(299,299)
boxes = detector.detect(frame)
 # lets's draw boxes, just multiply each predicted [0, 1] relative coordinate to image side in pixels respectively
for box in boxes:
        lx = int(round(box[0] * frame.shape[1]))
        ly = int(round(box[1] * frame.shape[0]))
        rx = int(round(box[2] * frame.shape[1]))
        ry = int(round(box[3] * frame.shape[0]))
        # x, y, w, h here
        ax.add_patch(cv2.rectangle((lx,ly),rx - lx,ry - ly,linewidth=2,edgecolor='r',facecolor='none'))


face_extraction.face_extract(storagepath=facestoragepath, path=framepath)

prediction.predict_video(facepath=facestoragepath)
