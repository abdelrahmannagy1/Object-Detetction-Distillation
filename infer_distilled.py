import sys
sys.path.append('utils/')
from keras.models import load_model
from image_preprocessing_ver1 import ImageDataGenerator
from keras.models import Model
from tqdm import tqdm
import cv2
from keras.layers import Lambda, concatenate, Activation,Dense
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K
from keras.losses import mean_squared_error as l2
from keras.losses import mean_absolute_error as l1
from mobilenet import get_mobilenet
from keras.applications.mobilenet import preprocess_input
import pandas as pd
import tensorflow as tf
import numpy as np

def IoU_Dataset(y_true, y_pred):

    
    xA = tf.maximum(y_pred[:,0],y_true[:,4])
    yA = tf.maximum(y_pred[:,1],y_true[:,5])
    xB = tf.maximum(y_pred[:,2],y_true[:,6])
    yB = tf.maximum(y_pred[:,3],y_true[:,7])
    
    interArea = tf.maximum(K.constant(0.0,shape=(1,)), xB - xA + 1) * tf.maximum(K.constant(0.0,shape=(1,)), yB - yA + 1)
    boxAArea = (y_pred[:,2] - y_pred[:,0] + 1) * (y_pred[:,3] - y_pred[:,1] + 1)
    boxBArea = (y_true[:,6] - y_true[:,4] + 1) * (y_true[:,7] - y_true[:,5] + 1)
    
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou
    
def IoU_Teacher(y_true, y_pred):
    tf.cast(
    y_true,
    tf.float32,
    name=None)
    
    tf.cast(
    y_pred,
    tf.float32,
    name=None)
    
    xA = tf.maximum(y_pred[:,0],y_true[:,0])
    yA = tf.maximum(y_pred[:,1],y_true[:,1])
    xB = tf.maximum(y_pred[:,2],y_true[:,2])
    yB = tf.maximum(y_pred[:,3],y_true[:,3])
    
    interArea = tf.maximum(K.constant(0.0,shape=(1,)), xB - xA + 1) * tf.maximum(K.constant(0.0,shape=(1,)), yB - yA + 1)
    boxAArea = (y_pred[:,2] - y_pred[:,0] + 1) * (y_pred[:,3] - y_pred[:,1] + 1)
    boxBArea = (y_true[:,2] - y_true[:,0] + 1) * (y_true[:,3] - y_true[:,1] + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def bounded_loss(rt, rs, m=0.5, v=0.5):
    """
        Input: Rs:     regression output from student network
               Rt:     regression output from teacher
               y_true: ground truth bounding box

        Output:Loss
    """
    #if(l2(rs,rt[1]) + m > l2(rt[0] , rt[1]) ):
        #lb = l2(rs,rt[1])
    #else:
        #lb = tf.constant(0,dtype=float)
    #lreg =l1(rs,rt[1]) + v*lb
    print(rt)
    #print(rs)
    #print(x)
    bound = l2(rs,rt[:, 4:]) + K.constant(m,shape=(1,))
    bound2 = l2(rt[:, :4] , rt[:, 4:])
    cond = tf.less(tf.reduce_mean(bound), tf.reduce_mean(bound2))
    #print(bound2.shape)
    
    #lb = tf.Variable(0.0, name="lb")
    #lreg = tf.Variable(0.0, name="lreg")
    #lreg = K.mean(K.square(rs - x[1]))
    #tf.cond(l2(rs,x[1]) + m > l2(x[0] , x[1]),lambda:tf.assign(lb,l2(rs,x[1])),lambda:tf.assign(lb,0.0))
    loss2 = lambda: l2(rs,rt[:, 4:])
    zero = lambda: K.constant(0.0,shape=(1,))
    res = tf.cond(cond,loss2,zero)
    return res


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def draw_box_on_image(boxes, im_width, im_height, image_np):
    
        
        

    image_np = cv2.resize(image_np,(im_width,im_height))
    (left, right, top, bottom) = (boxes[0] * im_width, boxes[2] * im_width,
                                    boxes[1] * im_height, boxes[3] * im_height)

    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
    cv2.imshow('view',image_np)


model = load_model('model',custom_objects={'<lambda>': bounded_loss,'IoU_Teacher':IoU_Teacher,'IoU_Dataset':IoU_Dataset})

data_dir = 'F:/distillation/OD_distill/'

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

test_generator = data_generator.flow_from_directory(
    data_dir + 'data/test',
    target_size=(224, 224),
    batch_size=2
)

predictions = model.predict_generator(test_generator,verbose=1,steps=1)
print(predictions)

im = cv2.imread("F:/distillation/OD_distill/data/test/hand/2.jpg")
draw_box_on_image(np.absolute(predictions[0]),600,400,im)

#for x_batch, _, name_batch in test_generator:
    #print(x_batch)
    #for i, n in enumerate(x_batch):
        #if(i == 0):
         #   draw_box_on_image(predictions[0],600,400,x_batch[0])
          #  if cv2.waitKey(25) & 0xFF == ord('q'):
           #     cv2.destroyAllWindows()
            #    break
        
