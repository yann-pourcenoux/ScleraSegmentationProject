import numpy as np
import os
import cv2
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.merge import *
from metrics import *
import matplotlib.pyplot as plt
from models import *
from tqdm import tqdm
from random import sample
from utils import *
from feedingData import *

""" Utils to predict and save predicted images """

def up_scaling(Y_pred, path_folder_masks, nb_to_display=5):
    """ up-scales predictions in order to visualize the results
    # Arguments
        Y_pred: array of predicted masks
        path_folder_masks: path of the folder containing the ground truth
        nb_to_display: number of images to keep in memory to visualize them

    # Returns
        The IoU value over the predictions
        The list of indexs to load the ground truth to visualize the results
        The list of up-scaled predicted masks
    """

    path_masks = [path_folder_masks + idx for idx in sorted(os.listdir(path_folder_masks))]

    length = len(path_masks)
    iou_array=np.zeros((length,1), dtype=np.float32)
    Y_pred_array = []

    idx_array = sorted(sample([i for i in range(length)], nb_to_display))

    for i, pred in tqdm(enumerate(Y_pred), total=len(path_masks)):
        true = cv2.imread(path_masks[i], cv2.IMREAD_COLOR)[:,:,::-1][:,:,:1].astype(np.bool)
        target_size = (true.shape[1], true.shape[0])

        pred = cv2.resize(pred, target_size, cv2.INTER_LINEAR)
        ret, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
        pred = np.expand_dims(pred, axis=-1)

        if i in idx_array:
            Y_pred_array.append(pred)

        iou_array[i]=iou_mask(true, pred)

    return np.mean(iou_array), idx_array, Y_pred_array

def up_scaling_and_save(path_folder_images, Y_pred, path_folder_pred='.'):
    """ up-scales the predicted masks and saves them in another folder with the same architecture as the images folder
    # Arguments
        path_folder_images: path of the folder containing the images
        Y_pred: array of predicted masks
        path_folder_pred: path of the folder where the up-scaled predictions will be saved
    """
    create_folder(path_folder_pred)

    filenames = sorted(os.listdir(path_folder_images))

    for i, pred in tqdm(enumerate(Y_pred), total=len(filenames)):
        filename = filenames[i]
        img = cv2.imread(os.path.join(path_folder_images, filename), cv2.IMREAD_COLOR)[:,:,::-1]

        target_size = (img.shape[1], img.shape[0])

        pred = cv2.resize(pred, target_size, cv2.INTER_LINEAR)
        ret, pred = cv2.threshold(pred, 0.5, 1, cv2.THRESH_BINARY)
        pred = np.expand_dims(pred, axis=-1)

        cv2.imwrite(os.path.join(path_folder_pred, filename), pred.astype(np.uint)*255)

    return


def main():
    model = load_model('../checkpoints/unet-BN-he_normal-64-relu-cbam-SSERBC.h5', custom_objects={'iou':iou})
    DATA_FOLDER = "../data_sclera_cropped"
    PRED_FOLDER = "../data_sclera_masks_predicted"
    for data in sorted(os.listdir(DATA_FOLDER)):
        path_pred = os.path.join(PRED_FOLDER, data)
        create_folder(path_pred)
        X = load_images(os.path.join(DATA_FOLDER, data), target_size=(128,128))
        Y_pred = model.predict(X, batch_size=32)
        up_scaling_and_save(os.path.join(DATA_FOLDER, data), Y_pred, os.path.join(PRED_FOLDER, data))


if __name__ == '__main__':
    main()
