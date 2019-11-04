from keras import backend as K
import tensorflow as tf
import numpy as np

""" Implementation of the IoU metric and a few utils to calculate
The IoU metric is used from this thread https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63044
"""


def castF(x):
    """ Shortcut function to cast array into float """
    return K.cast(x, K.floatx())

def castB(x):
    """ Shortcut function to cast array into bool """
    return K.cast(x, bool)

def iou_loss_core(true,pred):
    """ Computes the IoU over a batch
    # Arguments
        true: batch of ground truth
        pred: batch of predicted masks

    # Returns
        The IoU loss score of the batch
    """
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def iou(true, pred):
    """ IoU metric that can be used in keras
    # Arguments
        true: batch of ground truth masks
        pred: batch of predicted masks

    # Returns
        The average IoU value over the batch
    """

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thresholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])

def iou_mask(true, pred):
    """ my own implementation to calculate IoU over an image, really similar to iou_loss_core"""
    true = true.flatten().astype(np.float)
    pred = pred.flatten().astype(np.float)

    intersection = true*pred
    notTrue = 1-true
    union = true + (notTrue * pred)

    epsilon = 1e-7
    iou = (intersection.sum()+epsilon)/(union.sum()+epsilon)
    return iou

def iou_batch(true, pred, thresh=0.5):
    """ my own implementation to calculate IoU over a batch of masks """
    len_batch = true.shape[0]
    iou_array = np.zeros((len_batch,1),dtype=np.float)
    for i in range(len_batch):
        iou_array[i] = iou_mask(true[i], pred[i])

    iou_array = iou_array[iou_array>0.5]
    print(str(len_batch-iou_array.shape[0]) + " predicted masks had an IoU less than " + str(thresh))
    return np.mean(iou_array)
