
# TFLite for video classification model

import tensorflow as tf
from PIL import Image
import numpy

model = tf.lite.Interpreter(model_path="model.tflite")
classes = [  "Yuko" ,  "Wazari" ,  "Ipon" ,  ]

n_frames  =16

# Learn about its input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

model.resize_tensor_input(input_details[0]['index'], (1, n_frames,  172, 172, 3))
model.allocate_tensors()

video_numpy = numpy.ones((1, n_frames,  172, 172, 3)).astype('float32') #video numpy array at 5fps, pixels scaled from 0 to 1, with frame size 172, 172

model.set_tensor(input_details[0]['index'], video_numpy)
model.invoke()

class_scores = model.get_tensor(output_details[0]['index'])

print("")
print("class_scores", class_scores)
print("Class : ", classes[class_scores.argmax()])