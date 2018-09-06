import numpy as np
import sys
sys.path.append('utils/')
from keras.models import Sequential
import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Lambda, concatenate, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
import h5py
import matplotlib.pyplot as plt

h5f_t = h5py.File(SCRIPT_PATH+'/vggmodel_train/trainset.h5', 'r')
x_train = h5f_t['dataset_1'][:]
h5f_t.close()

h5f_t = h5py.File(SCRIPT_PATH+'/vggmodel_train/train_logits.h5', 'r')
train_logits = h5f_t['dataset_1'][:]
h5f_t.close()

h5f_t = h5py.File(SCRIPT_PATH+'/vggmodel_train/valset.h5', 'r')
x_val = h5f_t['dataset_1'][:]
h5f_t.close()

h5f_t = h5py.File(SCRIPT_PATH+'/vggmodel_train/val_logits.h5', 'r')
val_logits = h5f_t['dataset_1'][:]
h5f_t.close()


#data augmentation
datagen = ImageDataGenerator()  # randomly flip images
train_generator = datagen.flow(x_train,train_logits , batch_size=64)
val_generator = datagen.flow(x_val,val_logits , batch_size=64,shuffle=True)

# create student model
num_classes = 10
model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(96, activation='relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

temperature  = 5.0
# remove softmax
model.layers.pop()

# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)

# softed probabilities
logits_T = Lambda(lambda x: x/temperature)(logits)
probabilities_T = Activation('softmax')(logits_T)

output = concatenate([probabilities, probabilities_T])
model = Model(model.input, output)
# now model outputs 512 dimensional vectors

def knowledge_distillation_loss(y_true, y_pred, lambda_const):    
    y_true, logits = y_true[:,:10], y_true[:,10:]
    # convert logits to soft targets
    y_soft = K.softmax(logits/temperature)
    y_pred, y_pred_soft = y_pred[:, 10:], y_pred[:, :10]    
    return lambda_const*logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)
def accuracy(y_true, y_pred):
    y_true = y_true[:,:10]
    y_pred = y_pred[:, 10:]
    return categorical_accuracy(y_true, y_pred)
def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:,:10]
    y_pred = y_pred[:, 10:]
    return top_k_categorical_accuracy(y_true, y_pred)
def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:,:10]
    y_pred = y_pred[:, 10:]
    return logloss(y_true, y_pred)
def soft_logloss(y_true, y_pred):
    logits = y_true[:,10:]   
    y_soft = K.softmax(logits/temperature)
    y_pred_soft = y_pred[:, :10]    
    return logloss(y_soft, y_pred_soft)

lambda_const = 0.07
model.compile(
    optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True), 
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const), 
    metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
)

model.fit_generator(
    train_generator, 
    steps_per_epoch=400, epochs=30, verbose=1,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01), 
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007)
    ],
    validation_data=val_generator, validation_steps=80
    )


plt.plot(model.history.history['categorical_crossentropy'], label='train')
plt.plot(model.history.history['val_categorical_crossentropy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('logloss')
plt.show()

plt.plot(model.history.history['accuracy'], label='train')
plt.plot(model.history.history['val_accuracy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(model.history.history['top_5_accuracy'], label='train')
plt.plot(model.history.history['val_top_5_accuracy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('top5_accuracy')
plt.show()

val_generator = datagen.flow(x_test, val_logits, batch_size=64,shuffle=False)
print(model.evaluate_generator(val_generator_no_shuffle, 80))