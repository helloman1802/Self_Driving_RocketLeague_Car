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


# This is the path to the Tensorflow object detection API
sys.path.append("/home/malachi/.local/lib/python3.6/site-packages/tensorflow/models/research/object_detection/")


# Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util

from utils import visualization_utils as vis_util


# Model preparation 

# Variables
MODEL_NAME = 'ball_graph'

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
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def finder(show_image):
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 30, 'left': 0, 'width': 800, 'height': 600}
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
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          
          
          print('fps: {0}'.format(1 / (time.time()-last_time)))
          if show_image == 1:
            cv2.imshow('object detection', image_np)
            print('fps: {0}'.format(1 / (time.time()-last_time)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              break

finder(1)