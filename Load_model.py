from tensorflow.python.keras.layers import Conv2D, Input, ZeroPadding2D, Dense, Lambda
from tensorflow.python.keras.models import Model
from keras._tf_keras.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
tf.compat.v1.disable_eager_execution
import math
import numpy as np
import cv2
from keras._tf_keras.keras import Sequential
def load_mobilenetv2_224_075_detector(path):
  # Load the pre-trained MobileNetV2 model with the specified input shape and alpha value.
  # This ensures the model architecture matches the one used to train the weights you're loading.
  pre_trained_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), alpha=0.75)


  # Create a new model with the desired output layers.
  input_tensor = pre_trained_model.input  # Use the input layer from the pre-trained model
  output_tensor = pre_trained_model.output
  output_tensor = tf.keras.layers.ZeroPadding2D()(output_tensor)
  output_tensor = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

  model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

  # Load the weights, but only for the layers that match the pre-trained model.
  # Exclude the output layers since they are newly added and don't have pre-trained weights.
  for layer in pre_trained_model.layers:
    if layer.name in model.layers:
        model.get_layer(layer.name).set_weights(pre_trained_model.get_layer(layer.name).get_weights())

  return model




