from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D, Dense, Lambda
from tensorflow.python.keras.models import Model
from keras._tf_keras.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import numpy as np
import cv2
from keras._tf_keras.keras import Sequential
def load_mobilenetv2_224_075_detector(path):


  input_tensor = tf.keras.Input(shape=(224, 224, 3))
  output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
  output_tensor = tf.keras.layers.ZeroPadding2D()(output_tensor)
  output_tensor = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

  model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

  try:
    model.load_weights(path)
  except ValueError as e:
    # If the model architecture does not match the weight file architecture,
    # update the model architecture to match the weight file architecture.

    if 'You are trying to load a weight file containing 106 layers into a model with 1 layers.' in e:
      model.compile(optimizer='adam')

  return model




from tensorflow.python.keras.models import load_model
newmodel=load_model(input())