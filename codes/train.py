#!/usr/bin/env python
# coding: utf-8
import os
import sys
import math as m
from utils import *
from feedingData import *
from displayResults import *
from models import *
from metrics import *
from keras.callbacks import *

""" Train model on database """

def main():
    # Defining folders
    LOG_FOLDER = "../logs/"
    CHECKPOINT_FOLDER = "../checkpoints/"
    create_folder(LOG_FOLDER)
    create_folder(CHECKPOINT_FOLDER)

    # Loading data
    DATABASE = 'SSERBC_update'

    path_folder = os.path.join("../OldData/", DATABASE)

    path_folder_images = os.path.join(path_folder, 'images/')
    path_folder_masks = os.path.join(path_folder, 'masks/')

    path_images, idx_images = get_path_images(path_folder_images)
    path_masks, idx_masks = get_path_images(path_folder_masks)

    # Size of the images to be loaded in and that the network will process
    size = 128

    X, Y = load_data(path_folder_images, path_folder_masks, target_size=(size, size))

    # Defining model
    input_layer=Input((size,size,3))
    output_layer=build_unet(input_layer, start_depth=64)
    model = Model(inputs=input_layer, outputs=output_layer)

    metrics = [iou, 'accuracy']
    model.compile(
        optimizer=Adam(lr=1e-3),
        loss='binary_crossentropy',
        metrics=metrics)
    model.summary()

    # Defining callbacks
    filename = DATABASE + '.model.best.h5'
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_FOLDER, filename),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    earlystopper = EarlyStopper(
        monitor='val_loss',
        patience=15,
        verbose=0,
        mode='min')
    reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=m.sqrt(0.1),
        patience=5,
        mode='min',
        min_lr=1e-7)
    logger = CSVLogger(filename=os.path.join(LOG_FOLDER, filename))
    callbacks = [checkpointer, earlystopper, reducer, logger]

    # Training
    results = model.fit(
        x=X,
        y=Y,
        validation_split=0.1,
        epochs=5000,
        callbacks=callbacks,
        verbose=2)

    return results

if __name__ == '__main__':
    args = sys.argv
    init(args)
    results = main()
    display_training_raw(results, metrics)
