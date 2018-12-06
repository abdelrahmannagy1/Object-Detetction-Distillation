import numpy as np
import sys
sys.path.append('utils/')

import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# use non standard flow_from_directory
from image_preprocessing_ver2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception
from tqdm import tqdm
from keras.models import Model
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
#from  tensorflow.losses import absolute_difference as l1
#from  tensorflow.losses import mean_squared_error as l2
data_dir = 'F:/distillation/OD_distill/'

train_logits = np.load(data_dir + 'datatrain_logits.npy')[()]
val_logits = np.load(data_dir + 'dataval_logits.npy')[()]
print(train_logits[list(train_logits.keys())[0]].shape)

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

# note: i'm also passing dicts of logits
train_generator = data_generator.flow_from_directory(
    data_dir + 'data/train', train_logits,
    target_size=(224, 224),
    batch_size=1
)

val_generator = data_generator.flow_from_directory(
    data_dir + 'data/val', val_logits,
    target_size=(224, 224),
    batch_size=1
)



    

model = get_mobilenet(224, alpha=0.25, weight_decay=1e-5, dropout=0.1)

# remove softmax
model.layers.pop()

# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)

boxes = Dense(4)(probabilities)

model = Model(model.input, [boxes])

def accuracy(y_true,y_pred):
    return

def bounded_loss(rt, rs, m, v):
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
    #print(rt)
    #print(rs)
    #print(x)
    bound = l2(rs,rt[:, 4:]) + m
    bound2 = l2(rt[:, :4] , rt[:, 4:])
    #print(bound2.shape)
    
    #lb = tf.Variable(0.0, name="lb")
    #lreg = tf.Variable(0.0, name="lreg")
    #lreg = K.mean(K.square(rs - x[1]))
    #tf.cond(l2(rs,x[1]) + m > l2(x[0] , x[1]),lambda:tf.assign(lb,l2(rs,x[1])),lambda:tf.assign(lb,0.0))
    return K.switch(bound < bound2,l2(rs,rt[:, 4:]),K.constant(0,shape=(1,)))

def bounded_loss_nogt(rs, rt, m, v):
    """        when the dataset is not available
        Input: Rs:     regression output from student network
               Rt:     regression output from teacher

        Output:Loss
    """

    #tf.cond(((l2(rs,rt) < (l2(rs,rt) + m ))or (l2(rs,rt) > (l2(rs,rt) - m))),lambda:tf.assign(lb,0),lambda:tf.assign(lb,l2(rs,rt)))

    #if((l2(rs,rt) < (l2(rs,rt) + m ))or (l2(rs,rt) > (l2(rs,rt) - m))):
        #lb = 0
    #else:
        #lb = l2(rs,rt)
        
    
    return 5
def toyLoss(y_true, y_pred):
    print(y_true[1][3])
    print(y_pred)
    return tf.multiply(3.9,4.6)

model.compile(
    optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True), 
    loss=lambda y_true, y_pred:bounded_loss(y_true,y_pred,0.5,0.5)
    
)

model.fit_generator(
    train_generator, 
    steps_per_epoch=400, epochs=30, verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01), 
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)
