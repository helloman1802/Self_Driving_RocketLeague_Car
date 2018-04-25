import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
import time
from PIL import Image
import mss
import cv2
#import pyautogui as gui
from pynput.keyboard import Key, Controller

keyboard = Controller()


# This is the path to the Tensorflow object detection API
sys.path.append("/home/malachi/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/")


# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util

from utils import visualization_utils as vis_util


# Variables
MODEL_NAME = 'arrow_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


  

def finder(show_image):
  
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 70, 'left': 0, 'width': 1280, 'height': 720}
        t = 2001
        r = 0
        l = 0
        while True:
          last_time = time.time()
          image = sct.grab(monitor)
          image_np = np.array(image)
          image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
          image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
          #print('np array shape: {}'.format(np.shape(image_np)))
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          # Only visualize if the score meets this prereuisit.
          #if scores[0][1] > .5:
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          

          
          arrow_dict = {}
          

          
          for i, b in enumerate(boxes[0]):
            if classes[0][i] == 1:
              if scores[0][i] > 0.3:

                mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                
                apx_distance = round( (1-(boxes[0][i][3] - boxes[0][i][1]))**4, 3)
                arrow_dict[apx_distance] = [mid_x, mid_y, scores[0][i]]
                cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*1280), int(mid_y*720)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                x_move = mid_x - 0.5
                y_move = mid_y - 0.5
                
                get_to_x = x_move/0.5
              """
                if r == 1:
                  keyboard.press('w')
                  keyboard.press('d')
                  time.sleep(0.05)
                  keyboard.release('d')
                  keyboard.release('w')
                  r = 0
                elif l == 1:
                  keyboard.press('w')
                  keyboard.press('a')
                  time.sleep(0.05)
                  keyboard.release('a')
                  keyboard.release('w')
                  l = 0
                if l == 0 and r == 0:
                  if get_to_x > 0.05:
                    keyboard.press('w')
                    keyboard.press('d')
                    time.sleep(0.05)
                    keyboard.release('d')
                    keyboard.release('w')
                  elif get_to_x < -0.05:
                    keyboard.press('w')
                    keyboard.press('a')
                    time.sleep(0.05)
                    keyboard.release('a')
                    keyboard.release('w')
                  else:
                    keyboard.press('w')
                
                """
                

                
          """      
                if apx_distance <= 0.5:
                  if mid_x > 0.3 and mid_x < 0.7:
                    cv2.putText(image_np, 'Hitting ball', (int(mid_x*800)-50, int(mid_y*600)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
          """

          
            
          
          print('fps: {0}'.format(1 / (time.time()-last_time)))
          if show_image == 1:
            cv2.imshow('object detection', image_np)
            #print('fps: {0}'.format(1 / (time.time()-last_time)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break



finder(1)