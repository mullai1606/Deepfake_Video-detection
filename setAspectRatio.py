def setaspectratio( filepath ):
  from PIL import Image
  from numpy import size
  from numpy import asarray
  image  = Image.open(filepath )
  #image= cv2.imread(ff_image_path+"/"+imgvid[3]+"/key2.jpg")
  width  = image.size[0]
  height = image.size[1]
  #print(width,height)
  aspect = width / float(height)

  ideal_width = 1280
  ideal_height = 720
  if(width != ideal_width or height!= ideal_height):
   ideal_aspect = ideal_width / float(ideal_height)

   if aspect > ideal_aspect:
    # Then crop the left and right edges:
    new_width = int(ideal_aspect * height)
    offset = (width - new_width) / 2
    resize = (offset, 0, width - offset, height)
   else:
    # ... crop the top and bottom:
    new_height = int(width / ideal_aspect)
    offset = (height - new_height) / 2
    resize = (0, offset, width, height - offset)

   thu = image.crop(resize)
   thu=thu.resize((ideal_width, ideal_height), Image.Resampling.LANCZOS)
   #model.fit(X_train, y_train, epochs=10)


   ffff= asarray(thu)
   return ffff
  else:
   ffff= asarray(image)
   return ffff
