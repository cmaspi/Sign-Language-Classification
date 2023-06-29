import numpy as np
import dataset
import model as MODEL
import tensorflow as tf
import pickle

# Hyperparameters
lr = 1e-3
batch_size = 32
num_epochs = 100

lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.9, min_delta=1e-6)
###


model = MODEL.get_model()
model.summary()

Images, annotations = dataset.get_data()
perm = np.arange(annotations.shape[0])
np.random.shuffle(perm)


Images = Images[perm]
annotations = annotations[perm]


split = int(annotations.shape[0]*0.8)
x_train = Images[:split]
y_train = annotations[:split]

x_test = Images[split:]
y_test = annotations[split:]


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss='mse')
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
