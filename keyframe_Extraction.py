import pandas as pd
import numpy as np
import cv2
import fnmatch
import os
import glob
#Finding keyframes for extracted frames
# def keyFrameFinding(path):
#    diff=[]
#    imageA = cv2.imread( path+'/image'+str(1)+'.jpg')
#    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#    diff.append(cv2.absdiff(grayA, grayA))

#    for i in range(1,l):#loops till l-1. For 7 images 6 comparisons are needed
#       imageA = cv2.imread( path+'/image'+str(i)+'.jpg')
#       imageB = cv2.imread(path+'/image'+str(i+1)+'.jpg')
#       #print(i,":::",imageA.shape)
#       #print(i+1,":::",imageB.shape)
#       grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
#       grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
#       diff.append(cv2.absdiff(grayA, grayB))
#       #print(diff)
#    mn = np.mean(diff)
#    st_d = np.std(diff)
#       #print(mn,st_d)
#    a = 4
#    ts = mn + (a * st_d)
#     #print('The threshold==>',ts)
#    print("length of diff array:",len(diff))
#    a_fr = []#Creating an empty list
#    for i in range(len(diff)):
#       mn = np.mean(diff[i])#Calculating the mean for each frame
#       st_d = np.std(diff[i])
#       fr_ts = mn + (4*st_d)#Finding the threshold values for each frame/image
#       #print(i,fr_ts)
#       a_fr.append([i,fr_ts])
#    keyframes = []
#    for i,ac_tr in(a_fr):
#       if ac_tr >= ts:
#           #print(i,ac_tr)
#           keyframes.append(i)
#    print("No of key frames:", keyframes)
#    print("Length:",len(keyframes))
#    #Renaming
#    x=len(fnmatch.filter(os.listdir(path), 'key*.jpg'))
#    if(x==0):   #if no keyframe has been previously created, create keyframes
#      for i in range(0,len(keyframes)):
#         oldname=path+"/image"+str(keyframes[i])+".jpg"
#         newname=path+"/key"+str(i+1)+".jpg"
#         os.rename(oldname, newname)
#    #Check
#    print("check:", len(fnmatch.filter(os.listdir(path), 'key*.jpg'))    )



def keyFrameFinding(path):
    diff = []

    # Load the first image and check if it exists
    imageA_path = f"{path}/image1.jpg"
    imageA = cv2.imread(imageA_path)
    if imageA is None:
        raise FileNotFoundError(f"Image not found: {imageA_path}")

    # Convert the first image to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    diff.append(cv2.absdiff(grayA, grayA))  # Initialize diff with a dummy value

    # Get the total number of images (l-1 comparisons needed)
    image_files = sorted(fnmatch.filter(os.listdir(path), 'image*.jpg'))
    l = len(image_files)
    if l < 2:
        raise ValueError("Not enough images to calculate keyframes")

    # Loop through all consecutive image pairs
    for i in range(1, l):
        imageA_path = f"{path}/image{i}.jpg"
        imageB_path = f"{path}/image{i+1}.jpg"

        imageA = cv2.imread(imageA_path)
        imageB = cv2.imread(imageB_path)

        # Validate that images are loaded correctly
        if imageA is None or imageB is None:
            raise FileNotFoundError(f"Missing frame at index {i} or {i+1}")

        # Convert to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference
        diff.append(cv2.absdiff(grayA, grayB))

    # Calculate the threshold
    mn = np.mean(diff)
    st_d = np.std(diff)
    a = 4
    ts = mn + (a * st_d)

    print("Calculated threshold:", ts)

    # Identify keyframes based on the threshold
    keyframes = []
    for i, d in enumerate(diff):
        mn = np.mean(d)
        st_d = np.std(d)
        fr_ts = mn + (4 * st_d)
        if fr_ts >= ts:
            keyframes.append(i)

    print("Number of keyframes:", len(keyframes))
    print("Keyframes:", keyframes)

    # Rename keyframes
    x = len(fnmatch.filter(os.listdir(path), 'key*.jpg'))
    if x == 0:  # No keyframes exist, create them
        for idx, frame_index in enumerate(keyframes):
            oldname = f"{path}/image{frame_index}.jpg"
            newname = f"{path}/key{idx+1}.jpg"
            if os.path.exists(oldname):
                os.rename(oldname, newname)

    # Debug: Validate renamed keyframes
    print("Number of keyframes created:", len(fnmatch.filter(os.listdir(path), 'key*.jpg')))


def remove(path):
   removefilelist=glob.glob(path+"/**/image*.jpg", recursive=True) # all imgs(not keyframes) from all videos
   for filePath in removefilelist: 
      try:
          os.remove(filePath)
          #print(filePath)
      except OSError:
          print("Error while deleting file",filePath)
      