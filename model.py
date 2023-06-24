import tensorflow as tf
import keras as K
import keras.layers as KL
from keras.layers import Conv2D, GlobalAveragePooling2D, Input, MaxPooling2D, Dense
from keras.models import Model


def get_model():
    inp = Input(shape=(50, 50, 1))
    c1 = Conv2D(5, 3, activation='leaky_relu')(inp)
    c1_pool = MaxPooling2D(pool_size=(2,2))(c1)
    c2 = Conv2D(10, 3, activation='leaky_relu')(c1_pool)
    c3 = Conv2D(10, 3, activation='leaky_relu')(c2)
    c3_pool = MaxPooling2D(pool_size=(2,2))(c3)
    c4 = Conv2D(20, 3, activation='leaky_relu')(c3_pool)
    c4_gp = GlobalAveragePooling2D()(c4)
    l5 = Dense(20, 'softmax')(c4_gp)
    return Model(inputs=inp, outputs=l5)
