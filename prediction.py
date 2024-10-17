
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
#Handling Image data Generators
test_data_gen = ImageDataGenerator()
from Load_model import newmodel
import os
import pandas as pd
import numpy as np

def predict_video(facepath):
   whatdata=[]
   extractedfaces = facepath
   #facepath="/content/drive/MyDrive/sample-faces/"+vidname
   if os.path.exists(facepath):
      for filename in extractedfaces:
        if filename.endswith("jpg"):
          # Your code comes here such as
          whatdata.append(facepath+"/"+filename)
   print(whatdata)
   #creating DataFrame
   videodf = pd.DataFrame(whatdata, columns=['path'])
   videodf['path'] = videodf['path'].apply(lambda x: str(x))
   bs= len(videodf)
   print(bs)

   video_imgs = test_data_gen.flow_from_dataframe(dataframe=videodf,x_col='path', target_size=(299,299), class_mode=None,batch_size=bs, shuffle=False)

   video_prediction = newmodel.predict(video_imgs)

   predictclass=[]
   for i in range(0, len(video_prediction)):
     predictclass.append(  str(video_prediction[i][:2])  )
      # for value in predictclass:
      #  print(value)

   video_output = np.argmax(video_prediction, axis=1)
   print(video_output[0:25])
   avg = video_output.mean()
   print(avg)
   if avg >= 0.5 :
     predictedclass='real'
   elif avg <0.5:
     predictedclass='fake'

   print("PREDICTED :" +predictedclass)  


