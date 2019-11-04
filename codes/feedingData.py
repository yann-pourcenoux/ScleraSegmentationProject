import numpy as np
import sys
import cv2
from tqdm import tqdm
import os

def get_path_images(path_folder):
    """ Tool to get the names and paths of the files inside a folder
    # Arguments
        path_folder: string or POSIX path to the folder

    # Returns
        path_lis: i think the name is obvious enough
        idxs: list of the name of the files in the folder
    """
    idxs = sorted(os.listdir(path_folder))
    path_list = [os.path.join(path_folder, idx) for idx in idxs]
    return path_list, idxs

def load_data(path_folder_images, path_folder_masks, target_size=(128,128)):
    """ Loads the images and the masks
    # Arguments
        path_folder_images: path to the folder containing the images
        path_folder_masks: path to the folder containing the masks
        target_size: tuple of size 2 of the target size in which the images will be resized

    # Returns
        X_train: array of images
        Y_train: array of masks
    """
    assert len(os.listdir(path_folder_images))==len(os.listdir(path_folder_masks)), "Number of images and masks are different"
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()

    X_train = load_images(path_folder_images, target_size)
    Y_train = load_masks(path_folder_masks, target_size)

    return X_train, Y_train

def load_images(path_folder_images, target_size=(128,128)):
    """ Loads the images
    # Arguments
        path_folder_images: path to the folder containing the images
        target_size: tuple of size 2 of the target size in which the images will be resized

    # Returns
        X_train: array of images
    """
    path_images, *_ = get_path_images(path_folder_images)
    LEN_DATA = len(path_images)
    X_train = np.zeros((LEN_DATA, *target_size, 3), dtype=np.float32)

    for n, path_ in tqdm(enumerate(path_images), total=LEN_DATA):
        img = cv2.imread(path_, cv2.IMREAD_COLOR)[:,:,::-1]
        img = cv2.resize(img, target_size, cv2.INTER_AREA)

        X_train[n] = img.astype(np.float32)

    return X_train

def load_masks(path_folder_masks, target_size=(128,128)):
    """ Loads the images and the masks
    # Arguments
        path_folder_masks: path to the folder containing the masks
        target_size: tuple of size 2 of the target size in which the images will be resized

    # Returns
        Y_train: array of masks
    """
    path_masks, *_ = get_path_images(path_folder_masks)
    LEN_DATA = len(path_masks)
    Y_train = np.zeros((LEN_DATA, *target_size, 1), dtype=np.bool)

    for n, path_ in tqdm(enumerate(path_masks), total=LEN_DATA):
        mask = np.zeros((*target_size, 1), dtype=np.bool)
        mask_ = cv2.imread(path_, cv2.IMREAD_COLOR)[:,:,::-1][:,:,:1]
        mask_ = cv2.resize(mask_, target_size, cv2.INTER_AREA)
        mask_ = mask_[:,:,np.newaxis]
        mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    return Y_train
