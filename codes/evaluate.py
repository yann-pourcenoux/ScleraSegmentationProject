from utils import init, print_files
from predict import *
from feedingData import *
from displayResults import *
import os
from metrics import *
%matplotlib inline

init(['train.py', '0'])

path_folder = os.path.join('../OldData/SBVP_Sclera_Specific//')
path_folder_images = os.path.join(path_folder, 'images/')
path_folder_masks = os.path.join(path_folder, 'masks/')

X = load_images(path_folder_images, target_size=(128,128))

from keras.models import load_model
from metrics import iou
initializers = ['unet-BN-glorot_normal-64-relu-cbam-SSERBC.h5',
                'unet-BN-he_normal-64-relu-cbam-SSERBC.h5',
                'unet-BN-lecun_normal-64-relu-cbam-SSERBC.h5']
initializers = [os.path.join('../checkpoints/', init) for init in initializers]

for path in initializers:
    model = load_model(path, custom_objects={'iou':iou})
    Y_pred = model.predict(X, batch_size=32, verbose=2)
    val_iou, *_ = up_scaling(Y_pred, path_folder_masks, nb_to_display=0)
    print(path, str(val_iou))
