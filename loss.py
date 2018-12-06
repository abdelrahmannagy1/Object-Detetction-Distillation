# Defines the loss functions to minimize an object detection model
# 1- Classification error
# 2- Bounded loss for bounding box regression

from keras.losses import categorical_crossentropy as logloss
from keras.losses import mean_squared_error as l2
from keras.losses import mean_absolute_error as l1

def classification_loss(y_true, y_pred, mu,temp = 5):  
    # TODO Class weighted cross entropy(LogLoss) for
    # background and foreground classes
      
    
    # split in 
    #    onehot hard true targets
    #    logits from xception
    #    CHANGE THE SECOND INDEX
    y_true, logits = y_true[:, :256], y_true[:, 256:]
    
    # convert logits to soft targets
    y_soft = K.softmax(logits/temp)
    
    # split in 
    #    usual output probabilities
    #    probabilities made softer with temperature
    #    CHANGE THE SECOND INDEX
    y_pred, y_pred_soft = y_pred[:, :256], y_pred[:, 256:]    
    
    return mu*logloss(y_true, y_pred) + (1-mu)*logloss(y_soft, y_pred_soft)

def bounded_loss(rs, rt, y_true, m, v):
    """
        Input: Rs:     regression output from student network
               Rt:     regression output from teacher
               y_true: ground truth bounding box

        Output:Loss
    """
    if(l2(rs-y_true) + m > l2(rt - y_true) )
        lb = l2(r-y_true)
    else:
        lb = 0
    lreg =l1(rs,y_true) + v*lb
    return lreg

def bounded_loss_nogt(rs, rt, m, v):
    """
        when the dataset is not available
        Input: Rs:     regression output from student network
               Rt:     regression output from teacher

        Output:Loss
    """
    if((l2(rs-rt) < l2(rs-rt) + 5 )or (l2(rs-rt) > l2(rs-rt) - m)):
        lb = 0
    else:
        lb = l2(rs-rt)
        
    
    return lb


def hint_loss():
    # TODO
