import os
import shutil

import numpy as np
import cv2

def get_image(name, images_folder, predictions_folder):
    """returns a concatenate image of the eye image, the mask and the sclera region
    # Arguments
        name: name of the image
        images_folder: folder containing all the images of the cropepd eyes
        predictions_folder : folder containing all the predictions

    # Returns
        concat: concatenate image

    """
    person = 'data_' + name.split('_')[0]

    img_path = os.path.join(images_folder, person, name)
    pred_path = os.path.join(predictions_folder, person, name)

    img = cv2.imread(img_path)
    pred = cv2.imread(pred_path)
    sclera = img*pred.astype(np.bool)

    concat = np.concatenate((img, pred, sclera))
    return concat

def skip(name, good_folder, crop_folder, bad_folder):
    """ Retuns True if name is in the following folders """
    return (name in os.listdir(good_folder) or name in os.listdir(crop_folder) or name in os.listdir(bad_folder))

if __name__ == '__main__':
    # Defining folders
    PREDICT_FOLDER = '../predictions'
    IMAGES_FOLDER = '../data_sclera_cropped/'

    GOOD_FOLDER = '../data_sclera_sorted/good'
    BAD_FOLDER = '../data_sclera_sorted/bad'
    CROP_FOLDER = '../data_sclera_sorted/crop'

    fQUIT = 0
    key = 0
    
    # Instructions
    print("Press A to save in GOOD_FOLDER, Z to CROP_FOLDER to crop again, E to BAD_FOLDER, R to go back and finally Q to quit")

    # Going through every images
    for person in sorted(os.listdir(PREDICT_FOLDER)):
        path_person = os.path.join(PREDICT_FOLDER, person)
        imgs = sorted(os.listdir(path_person))
        i = 0
        while i<len(imgs):
            img_name=imgs[i]
            while skip(img_name, GOOD_FOLDER, CROP_FOLDER, BAD_FOLDER) and key != 114:
                i+=1
                if i == len(imgs):
                    break
                img_name=imgs[i]

            concat=get_image(img_name, IMAGES_FOLDER, PREDICT_FOLDER)

            cv2.namedWindow("Images", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("Images", concat)
            key = cv2.waitKey(0)

            if key == 97:
                shutil.copy(os.path.join(path_person, img_name), GOOD_FOLDER)

            elif key == 122:
                shutil.copy(os.path.join(path_person, img_name), CROP_FOLDER)

            elif key == 101:
                shutil.copy(os.path.join(path_person, img_name), BAD_FOLDER)

            elif key == 114:
                i -=2

            elif key == 113:
                fQUIT = 1
                break

            if fQUIT == 1:
                break

            i+=1

        if fQUIT == 1:
            break

    print("FINITO")
