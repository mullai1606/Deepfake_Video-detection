import pandas as pd
import numpy as np
import cv2
import fnmatch
import os
import glob
#Finding keyframes for extracted frames
def keyFrameFinding(path):
   diff=[]
   imageA = cv2.imread( path+'/image'+str(1)+'.jpg')
   grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
   diff.append(cv2.absdiff(grayA, grayA))

   for i in range(1,l):#loops till l-1. For 7 images 6 comparisons are needed
      imageA = cv2.imread( path+'/image'+str(i)+'.jpg')
      imageB = cv2.imread(path+'/image'+str(i+1)+'.jpg')
      #print(i,":::",imageA.shape)
      #print(i+1,":::",imageB.shape)
      grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
      grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
      diff.append(cv2.absdiff(grayA, grayB))
      #print(diff)
   mn = np.mean(diff)
   st_d = np.std(diff)
      #print(mn,st_d)
   a = 4
   ts = mn + (a * st_d)
    #print('The threshold==>',ts)
   print("length of diff array:",len(diff))
   a_fr = []#Creating an empty list
   for i in range(len(diff)):
      mn = np.mean(diff[i])#Calculating the mean for each frame
      st_d = np.std(diff[i])
      fr_ts = mn + (4*st_d)#Finding the threshold values for each frame/image
      #print(i,fr_ts)
      a_fr.append([i,fr_ts])
   keyframes = []
   for i,ac_tr in(a_fr):
      if ac_tr >= ts:
          #print(i,ac_tr)
          keyframes.append(i)
   print("No of key frames:", keyframes)
   print("Length:",len(keyframes))
   #Renaming
   x=len(fnmatch.filter(os.listdir(path), 'key*.jpg'))
   if(x==0):   #if no keyframe has been previously created, create keyframes
     for i in range(0,len(keyframes)):
        oldname=path+"/image"+str(keyframes[i])+".jpg"
        newname=path+"/key"+str(i+1)+".jpg"
        os.rename(oldname, newname)
   #Check
   print("check:", len(fnmatch.filter(os.listdir(path), 'key*.jpg'))    )

def remove(path):
   removefilelist=glob.glob(path+"/**/image*.jpg", recursive=True) # all imgs(not keyframes) from all videos
   for filePath in removefilelist: 
      try:
          os.remove(filePath)
          #print(filePath)
      except OSError:
          print("Error while deleting file",filePath)
      