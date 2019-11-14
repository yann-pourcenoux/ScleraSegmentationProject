# Sclera Segmentation Project

I worked on this project when I was an intern in the Computer Vision Laboratory in the Faculty of Computer Sciences in Lubljana (Slovenia) in summer 2019. This project is part of a bigger one which aims at recognizing people based on their sclera vessels.
This project was my first contact with deep learning, CNNs and the related librairies (keras and tensorflow).
The full report can be sent to you if you e-mail me at yann.pourcenoux@gmail.com.

### Prerequisites

To run this codes, you will need the common python librairies as well as [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) (The GPU-versions will make the computation faster).

## Project

The goal was to be able to isolate the sclera area from pictures wherever the person is looking at, and achieving the best accuracy whatever the camera used.
In the codes section not only the codes for the model can be found but also the codes for loading and visualizing both the data and the predictions of the model.

### Data collection

Data from 100 people have been gathered with multiple camera and with subjects looking at different orientations in several lighting conditions. 

![Different Orientations](https://github.com/En3rg1/ScleraSegmentationProject/tree/master/figures/orientations.png)

### Model

The model implemented is a [Unet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) particularly popular for medical imaging. This model has been improved with a bottleneck after all the informations have been encoded to reach better performances.

The final models reaches an IoU around 90% on the training dataset. 

The predictions are quite good on the data collected for the project but it struggles in some cases, mainly in low-lighting conditions.

[Predictions](https://github.com/En3rg1/ScleraSegmentationProject/tree/master/figures/predictions.png)

## Future work
The code would need to be updated since the framework tensorflow.keras in tensorflow 2.0 is more efficient.
Since the model struggles in low luminosity and that histogram equalization didn't improve the results. Another preprocessing method may be then the way to reach better performances.


## Acknowledgments

I am thankfull to everyone from the Computer Vision Lab in Ljubljana who gave me the opportunity to work on this project and helped me all along the project.

