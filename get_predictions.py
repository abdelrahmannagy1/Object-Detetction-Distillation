from tqdm import tqdm
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
sys.path.append('utils/')

# use non standard flow_from_directory
from image_preprocessing_ver1 import ImageDataGenerator
# it outputs not only x_batch and y_batch but also image names

from keras.models import Model

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util

data_dir = "./data"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = './TeacherModel/' + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('./TeacherModel/', 'hand_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#DATASET TARGETS
T = pd.read_csv('./data/train/hands/hands_labels.csv')
V = pd.read_csv('./data/test/hands/hands_labels.csv')


def preprocess_input(x):
  pass
    #x /= 255.0
    #x -= 0.5
    #x *= 2.0
    #return x

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            #print(output_dict['detection_boxes'])
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    data_dir + '/train/', 
    target_size=(299, 299),
    batch_size=1, shuffle=False
)

val_generator = data_generator.flow_from_directory(
    data_dir + '/test/', 
    target_size=(299, 299),
    batch_size=1, shuffle=False
)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
v_gt = {}
for i, row in tqdm(V.iterrows()):
    v_gt[row.filename] = [row.xmin,row.ymin,row.xmax,row.ymax]

t_gt = {}
for i, row in tqdm(T.iterrows()):
    t_gt[row.filename] = [row.xmin,row.ymin,row.xmax,row.ymax]



batches = 0
train_logits = {}

for x_batch, _, name_batch in tqdm(train_generator):
    print(name_batch)
    squeezed = np.squeeze(x_batch,axis=0)
    
    batch_logits = run_inference_for_single_image(squeezed,detection_graph)
    #print(batch_logits)
    print(batch_logits['detection_boxes'].shape)
    for i, n in enumerate(name_batch):
        print(n)
        train_logits[n] =np.array(batch_logits['detection_boxes'][0])
        #train_logits[n] = train_logits[n].transpose()
        ##train_logits[n].shape = (1,)
        print(train_logits[n].shape)
        
        x = n.split("\\")
        #print(x)
        #print(t_gt[x[-1]])
        t_gt[x[-1]][0] /= 1280
        t_gt[x[-1]][1] /= 720
        t_gt[x[-1]][2] /= 1280
        t_gt[x[-1]][3] /= 720
        train_logits[n]= np.append( train_logits[n],(t_gt[x[-1]]))
        #train_logits[n].flatten(order='A')
        #train_logits[n]=np.reshape(train_logits[n],(1,))
        #train_logits[n]= train_logits[n].transpose()
        print (train_logits[n].shape)
    
    batches += 1
    if batches >= (4383): # 13662
        break

print(train_logits)

batches = 0
val_logits = {}

for x_batch, _, name_batch in tqdm(val_generator):
    
    squeezed = np.squeeze(x_batch,axis=0)
    #load_image_into_numpy_array(squeezed)
    batch_logits = run_inference_for_single_image(squeezed,detection_graph)
    
    for i, n in enumerate(name_batch):
        val_logits[n] = np.array(batch_logits['detection_boxes'][0])
        x = n.split("\\")
        v_gt[x[-1]][0] /= 1280
        v_gt[x[-1]][1] /= 720
        v_gt[x[-1]][2] /= 1280
        v_gt[x[-1]][3] /= 720
        val_logits[n]=np.append(val_logits[n],v_gt[x[-1]])
        #val_logits[n]=np.reshape(val_logits[n],(1,8))
        print (val_logits[n].shape)
        
    
    batches += 1
    if batches >= (399) : # 1222
        break

np.save(data_dir + 'train_logits.npy', train_logits)
np.save(data_dir + 'val_logits.npy', val_logits)
