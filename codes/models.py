import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.pooling import *
from keras.layers.merge import *
from keras.optimizers import *
from metrics import *

""" Implementation of U-net: https://arxiv.org/pdf/1505.04597.pdf
with Batch Normalization: https://arxiv.org/pdf/1502.03167.pdf
with CBAM attention block from the publication : https://arxiv.org/pdf/1807.06521.pdf
implemented by kobiso on : https://github.com/kobiso/CBAM-keras
The default initializer is he_normal from: https://arxiv.org/pdf/1502.01852.pdf

"""


def Conv2D_BN(x, filters, kernel_size, strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel_size in `Conv2D`.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        activation: activation in `Conv2D`.
        kernel_initializer: kernel_initializer in `Conv2D`.
        kernel_regularizer: kernel_regularizer in `Conv2D`.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def Conv2DTranspose_BN(x, filters, kernel_size, strides=(1,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2DTranspose`.
        kernel_size: kernel_size in `Conv2DTranspose`.
        padding: padding mode in `Conv2DTranspose`.
        strides: strides in `Conv2DTranspose`.
        activation: activation in `Conv2DTranspose`.
        kernel_initializer: kernel_initializer in `Conv2DTranspose`.
        kernel_regularizer: kernel_regularizer in `Conv2DTranspose`.

    # Returns
        Output tensor after applying `Conv2DTranspose` and `BatchNormalization`.
    """

    x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""

	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):

	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)

	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)

	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7

	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature

	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature._keras_shape[-1] == 1

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature])

def build_unet(input_layer = Input((128,128,3)), start_depth=64, activation='relu', initializer='he_normal'):
    """ The implementation of the U-net architecture described at the top of the file.
    In our case we got the best results with `start_depth=64` this is why it is the default setting
    """

    # 128 -> 64
    conv1 = Conv2D_BN(input_layer, start_depth * 1, (3, 3), activation=activation, kernel_initializer=initializer)
    conv1 = Conv2D_BN(conv1, start_depth * 1, (3, 3), activation=activation, kernel_initializer=initializer)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # 64 -> 32
    conv2 = Conv2D_BN(pool1, start_depth * 2, (3, 3), activation=activation, kernel_initializer=initializer)
    conv2 = Conv2D_BN(conv2, start_depth * 2, (3, 3), activation=activation, kernel_initializer=initializer)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # 32 -> 16
    conv3 = Conv2D_BN(pool2, start_depth * 4, (3, 3), activation=activation, kernel_initializer=initializer)
    conv3 = Conv2D_BN(conv3, start_depth * 4, (3, 3), activation=activation, kernel_initializer=initializer)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # 16 -> 8
    conv4 = Conv2D_BN(pool3, start_depth * 8, (3, 3), activation=activation, kernel_initializer=initializer)
    conv4 = Conv2D_BN(conv4, start_depth * 8, (3, 3), activation=activation, kernel_initializer=initializer)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # Middle
    convm=cbam_block(pool4)

    # 8 -> 16
    deconv4 = Conv2DTranspose(convm, start_depth * 8, (3, 3), strides=(2, 2), activation=activation, kernel_initializer=initializer)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Conv2D_BN(uconv4, start_depth * 8, (3, 3), activation=activation, kernel_initializer=initializer)
    uconv4 = Conv2D_BN(uconv4, start_depth * 8, (3, 3), activation=activation, kernel_initializer=initializer)

    # 16 -> 32
    deconv3 = Conv2DTranspose(uconv4, start_depth * 4, (3, 3), strides=(2, 2), activation=activation, kernel_initializer=initializer)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D_BN(uconv3, start_depth * 4, (3, 3), activation=activation, kernel_initializer=initializer)
    uconv3 = Conv2D_BN(uconv3, start_depth * 4, (3, 3), activation=activation, kernel_initializer=initializer)

    # 32 -> 64
    deconv2 = Conv2DTranspose(uconv3, start_depth * 2, (3, 3), strides=(2, 2), activation=activation, kernel_initializer=initializer)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D_BN(uconv2, start_depth * 2, (3, 3), activation=activation, kernel_initializer=initializer)
    uconv2 = Conv2D_BN(uconv2, start_depth * 2, (3, 3), activation=activation, kernel_initializer=initializer)

    # 64 -> 128
    deconv1 = Conv2DTranspose(uconv2, start_depth * 1, (3, 3), strides=(2, 2), activation=activation, kernel_initializer=initializer)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D_BN(uconv1, start_depth * 1, (3, 3), activation=activation, kernel_initializer=initializer)
    uconv1 = Conv2D_BN(uconv1, start_depth * 1, (3, 3), activation=activation, kernel_initializer=initializer)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)

    return output_layer

def get_unet(start_depth=64, size=128):
    """ create our U-Net model
    # Arguments
        start_depth: start_depth in `build_unet`
        size: size in `build_unet`

    # Returns
        model: U-Net model for one class compiled Adam optimizer and using IoU metric
    """

    input_layer = Input((size, size, 3))
    output_layer = build_unet(input_layer, start_depth)

    model = Model(inputs=input_layer, outputs=output_layer)

    metrics = [iou, 'accuracy']

    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=metrics)
    model.summary()

    return model
