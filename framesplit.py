#Frame split-up module

import cv2
import os
import shutil

def getFrame(sec, count, vidcap):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image) # save frame as JPG file in the current directory
        #plt.imshow(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.imshow(image)
    return hasFrames

def split(inputvidpath, vidname, path):
    vidcap = cv2.VideoCapture(inputvidpath)
    sec = 0
    frameRate = 0.10 #10 frames per second
    count=1
    vidname= inputvidpath.split("/")[5]
    framepath = path+"/splitFrames/"
    #path="/content/drive/MyDrive/sample-images/"+vidname
    if os.path.exists(framepath):
       shutil.rmtree(framepath, ignore_errors=True); #if path already exists delete it
    os.makedirs(framepath)
    os.chdir(framepath)#change the current directory to the path
    success = getFrame(sec, count,vidcap)
    while success:
       count = count + 1
       sec = sec + frameRate
       sec = round(sec, 2)
       success = getFrame(sec)
