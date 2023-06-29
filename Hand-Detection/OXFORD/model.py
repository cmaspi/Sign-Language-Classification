from keras.layers import Conv2D, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model
import keras

def get_model():
    inp = Input(shape=(200, 200, 3))
    c1 = Conv2D(2, 3, activation='leaky_relu')(inp)
    c1_pool = MaxPooling2D(pool_size=(3, 3))(c1)
    c2 = Conv2D(4, 3, activation='leaky_relu')(c1_pool)
    c2_pool = MaxPooling2D(pool_size=(3, 3))(c2)
    c3 = Conv2D(8, 3, activation='leaky_relu')(c2_pool)
    c3_pool = MaxPooling2D(pool_size=(3, 3))(c3)
    c4 = Conv2D(16, 3, activation='leaky_relu')(c3_pool)
    c4_f = Flatten()(c4)

    l1 = Dense(24, activation='leaky_relu')(c4_f)
    # l2 = Dense(50, activation='leaky_relu')(l1)
    l3 = Dense(4, activation='sigmoid')(l1)
    return Model(inputs=inp, outputs=l3)
