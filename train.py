import numpy as np
import dataset
import os
import model as MODEL
import tensorflow as tf
import pickle

# Hyperparameters
lr = 1e-3
batch_size = 32
num_epochs = 200

lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5)
###


model = MODEL.get_model()
model.summary()

Images, labels = dataset.get_random_uniform(150)
Images = Images[..., np.newaxis]/255
perm = np.arange(labels.size)
np.random.shuffle(perm)

print(Images.shape, perm)

Images = Images[perm]
labels = labels[perm]


split = int(labels.size*0.8)
x_train = Images[:split]
y_train = labels[:split]

x_test = Images[split:]
y_test = labels[split:]


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy())
history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=[x_test, y_test],
                    batch_size=batch_size,
                    callbacks=[lr_on_plateau],
                    verbose=1,
                    epochs=num_epochs)
model.save('chks/SavedModel_1')

with open('logs/model_log.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
