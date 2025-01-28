# MIT License

# Copyright (c) 2020 AISyLab @ TU Delft


#
# brief     : Auto encoder architecture
# author    : AISY lab & modified by Lucas D. Meier
# date      : 2020 / 2022
# copyright : publicly available on https://github.com/AISyLab/Denoising-autoencoder
# 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K

####################### Denoiser building blocks #######################
def _conv(x, filter_num, window_size, act, max_pool, dp_rate = 0):
  y = Conv1D(filter_num, window_size, padding='same')(x)
  y = BatchNormalization()(y)
  y = Activation(act)(y)
  if max_pool > 0:
    y = MaxPooling1D(max_pool)(y)
  if dp_rate > 0:
    y = Dropout(dp_rate)(y)
  return y

def _Conv1DTranspose(input_tensor, filters, kernel_size, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def _deconv(x, filter_num, window_size, act, max_pool, dp_rate = 0):
  if max_pool > 0:
    y = UpSampling1D(max_pool)(x)
  else:
    y = x
  y = _Conv1DTranspose(y, filter_num, window_size)
  y = BatchNormalization()(y)
  y = Activation(act)(y)

  if dp_rate > 0:
    y = Dropout(dp_rate)(y)
  return y


POOL_STRIDE_EXT = 5
POOL_STRIDE_INT = 2
MODEL_CONSTANT = POOL_STRIDE_EXT * POOL_STRIDE_INT * POOL_STRIDE_INT
def old_cnn(lr, input_length):
    """old_cnn : autoencoder architecture from :footcite:t:`[17]wu2020remove`. 
    
    The input length must have been padded to a multiple of 20. 
    
    
    .. footbibliography::

    """
    # assert input_length % 20 == 0
    img_input = Input(shape=(input_length, 1))
    #encoder
    x = _conv(img_input, 256, 2, 'selu', 0)
    x = _conv(x, 256, 2, 'selu', 0)
    x = _conv(x, 256, 2, 'selu', 0)
    x = _conv(x, 256, 2, 'selu', 0)
    x = _conv(x, 256, 2, 'selu', 0)
    x = _conv(x, 256, 2, 'selu', POOL_STRIDE_EXT)
    x = _conv(x, 128, 2, 'selu', 0)
    x = _conv(x, 128, 2, 'selu', 0)
    x = _conv(x, 128, 2, 'selu', 0)
    x = _conv(x, 128, 2, 'selu', POOL_STRIDE_INT)
    x = _conv(x, 64, 2, 'selu', 0)
    x = _conv(x, 64, 2, 'selu', 0)
    x = _conv(x, 64, 2, 'selu', 0)
    x = _conv(x, 64, 2, 'selu', POOL_STRIDE_INT)
    x = Flatten(name='flatten')(x)

    x = Dense(512, activation='selu')(x)  #middle of the model, 512 neurons -> MAY BE CHANGED

    x = Dense((input_length // MODEL_CONSTANT) * 64, activation='selu')(x) # 2240
    x = Reshape((input_length // MODEL_CONSTANT, 64))(x) #35
    x = _deconv(x, 64, 2, 'selu', POOL_STRIDE_INT)
    x = _deconv(x, 64, 2, 'selu', 0)
    x = _deconv(x, 64, 2, 'selu', 0)
    x = _deconv(x, 64, 2, 'selu', 0)
    x = _deconv(x, 128, 2, 'selu', POOL_STRIDE_INT)
    x = _deconv(x, 128, 2, 'selu', 0)
    x = _deconv(x, 128, 2, 'selu', 0)
    x = _deconv(x, 128, 2, 'selu', 0)
    x = _deconv(x, 256, 2, 'selu', POOL_STRIDE_EXT)
    x = _deconv(x, 256, 2, 'selu', 0)
    x = _deconv(x, 256, 2, 'selu', 0)
    x = _deconv(x, 256, 2, 'selu', 0)
    x = _deconv(x, 256, 2, 'selu', 0)
    x = _deconv(x, 256, 2, 'selu', 0)
    
    x = _deconv(x, 1, 2, 'sigmoid', 0)

    model = Model(img_input, x)
    #model = multi_gpu_model(model)
    opt = RMSprop(lr=lr)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #model.summary()
    return model