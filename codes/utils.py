import sys
import os
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def init(args=['train.py']):
    """ Initializes the right GPU and create TF session which prevents a few problems
    # Arguments
        args: list of the argumetns when running program
    """

    # Choosing which gpu to use and how many threads
    GPU = "0"
    X = "8" # default value -> using main GPU (here 1080ti)
    if len(args) > 1:
        GPU = args[1]

    if GPU == "1":
        X = "6" # need to change variable to use 1050ti

    os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]=X
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"]= GPU

    # Checking if the wanted GPUs are running
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    # Checking
    print(get_available_gpus())

    # Chossing the number of threads used
    THREADS = 4
    if len(args)>2:
        THREADS = args[2]

    # CUDNN failed to initialize ERROR fix
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def print_files(path_folder):
    """ print files in the folder from path_folder """
    print("The files in " + path_folder + " are :")
    for file in os.listdir(path_folder):
        print(file)

def create_folder(path):
    """ checks if folder exists and if not it creates it """
    if not os.path.exists(path):
        os.makedirs(path)
