from keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np
from metrics import *

from random import sample

def get_metrics_names(metrics):
    """ Function to get the metrics names from the metric list in order to display the results
    # Arguments
        metrics: list of the metrics used for training

    # Returns
        metrics_names: list of the metrics names
    """
    metrics_names = ['loss'] #add loss as first metric
    for metric in metrics:
        string = str(metric)
        string = string.split(' ')
        if len(string) > 1:
            string = string[1]
        else:
            string = string[0]
            if string == 'accuracy':
                string = 'acc'
        metrics_names.append(string)
    return metrics_names

def display_training_raw(results, metrics=['accuracy']):
    """ Displays the evolution of the metrics during training
    # Arguments
        results: History object of the training
        metrics: list of metrics to display
    """
    metrics_names = get_metrics_names(metrics)
    lines = len(metrics_names)
    fig, axes = plt.subplots(lines, 1, figsize=(5*lines, 15))
    for i in range(lines):
        metric = metrics_names[i]

        axes[i].plot(results.epoch, results.history[metric], label="Train "+metric)
        axes[i].plot(results.epoch, results.history["val_"+metric], label="Validation "+metric)

    plt.show(fig)
    return

def smooth_curve(points, factor=0.75):
    """ Smooth the list of points to get rid of the high variation
    # Arguments
        points: list of points
        factor: factor relative the smoothing efficiency
                    'new_point[i] = points[i-1]*factor + points[i]*(1-factor)'

    # Returns
        smoothed_points: list of points of the smoothed curve
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def display_training_smooth(results, metrics=["accuracy"], factor=0.75):
    """ Displays the smoothed evolution of the metrics during training
    # Arguments
        results: History object of the training
        metrics: list of metrics to display
        factor: coefficient used in 'smooth_curve'
    """
    metrics_names = get_metrics_names(metrics)
    lines = len(metrics)
    fig, axes = plt.subplots(lines, 1, figsize=(5*lines, 15))

    for i in range(lines):
        metric = metrics_names[i]

        axes[i].plot(results.epoch, smooth_curve(results.history[metric], factor), label="Train "+metric)
        axes[i].plot(results.epoch, smooth_curve(results.history["val_"+metric], factor), label="Validation "+metric)

    plt.show(fig)
    return

def display_predict_results(X, Y_pred, Y, number_of_images=5, num_class=1):
    """
    Displays the results from prediction
    For each class and each image it displays the IoU value
    and the image in which the Green area stands for True Positive pixels, the Red area stands for False Positives pixels and the Blue for False Negatives
    rows, cols = number_of_images, num_class+1

    # Arguments
        X: array of images used for prediction
        Y_pred: the array result of the prediction
        Y: the ground truth of the images used for predictions
        number_of_images: number of images to display
        num_class: number of class in the problem/data
    """
    
    idxs = sorted(sample([i for i in range(len(X))], number_of_images))
    images = []
    preds = []
    gts = []
    for idx in idxs:
        images.append(X[idx])
        preds.append(Y_pred[idx])
        gts.append(Y[idx])

    green = [107,142,35]
    red = [220,20,60]
    blue = [0,0,142]

    print('Displaying images and masks :')
    print('Green area stands for True Positive pixels, Red area stands for False Positives pixels and blue for False Negatives :')

    for i in range(rows):
        n=1
        plt.subplot(1,cols,n)
        plt.imshow(images[i])

        for j in range(num_class):
            gt = gts[i][:,:,j]
            pred = preds[i][:,:,j]>0.5

            iou = iou_mask(gt, pred)

            TP = np.multiply(gt, pred)
            FP = np.bitwise_xor(pred, TP)
            FN = np.bitwise_xor(gt, TP)

            img = np.zeros(X[idxs[i]].shape, 'uint8')

            img[TP]=green
            img[FP]=red
            img[FN]=blue

            n+=1
            plt.subplot(1,cols,n)
            plt.imshow(img)
            plt.title(str(iou))
        plt.show()
    return
