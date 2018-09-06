
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
import h5py
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import os
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0001
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        print('start build')
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights(SCRIPT_PATH+'/vggmodel_train/cifar10vgg.h5')
    def return_model(self):
        return self.model
    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        # model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # model.add(Dropout(0.25))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 3
        # learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('x_train shape')

        y_train = keras.utils.to_categorical(y_train,10)
        y_test = keras.utils.to_categorical(y_test, 10)
        print('y_train shape:', y_train.shape)
        y_train = y_train.reshape(-1,10)
        y_test = y_test.reshape(-1,10)
        print('y_train shape:', y_train.shape)

        # def lr_scheduler(epoch):
        #     return learning_rate * (0.5 ** (epoch // lr_drop))
        # reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # #data augmentation
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        # # (std, mean, and principal components if ZCA whitening is applied).
        # datagen.fit(x_train)

        # #optimization details
        # sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # # training process in a for loop with learning rate drop every 25 epoches.

        # historytemp = model.fit_generator(datagen.flow(x_train, y_train,
        #                                  batch_size=batch_size),
        #                     steps_per_epoch=x_train.shape[0] // batch_size,
        #                     epochs=maxepoches,
        #                     validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)

        #optimization details
        sgd = SGD(lr=0.0005, decay=0, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(x_train, y_train,
          batch_size=64,
          epochs=3,
          verbose=1,
          validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('test score = ',score)
        model.save_weights(SCRIPT_PATH+'/vggmodel_train/cifar10vgg.h5')
        return model

def extract_feature(model):
    from keras.models import Model
    from keras import layers
    from tqdm import tqdm

    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)


    #data augmentation
    datagen = ImageDataGenerator()  # randomly flip images
    train_generator = datagen.flow(x_train, y_train, batch_size=64)
    val_generator = datagen.flow(x_test, y_test, batch_size=64)

    batches = 0
    train_logits = []
    train_set = []
    for x_batch, name_batch in train_generator:
        batch_logits = model.predict_on_batch(x_batch)
        for i, n in enumerate(name_batch):
            train_logits.append(np.concatenate((batch_logits[i],n[0]),axis=0))
            train_set.append(x_batch[i])
        batches += 1
        if batches >= 400:
            break
        # ii += 1
    batches = 0
    val_logits = []
    val_set = []
    for x_batch, name_batch in val_generator:
        
        batch_logits = model.predict_on_batch(x_batch)
        
        for i, n in enumerate(name_batch):
            val_logits.append(np.concatenate((batch_logits[i],n[0]),axis=0))
            val_set.append(x_batch[i])
        
        batches += 1
        if batches >= 80:
            break

    h5f_t = h5py.File(SCRIPT_PATH+'/vggmodel_train/trainset.h5', 'w')
    h5f_t.create_dataset('dataset_1', data=train_set)
    h5f_t.close()


    h5f = h5py.File(SCRIPT_PATH+'/vggmodel_train/train_logits.h5', 'w')
    h5f.create_dataset('dataset_1', data=train_logits)
    h5f.close()

    h5f2_v = h5py.File(SCRIPT_PATH+'/vggmodel_train/valset.h5', 'w')
    h5f2_v.create_dataset('dataset_1', data=val_set)
    h5f2_v.close()


    h5f2 = h5py.File(SCRIPT_PATH+'/vggmodel_train/val_logits.h5', 'w')
    h5f2.create_dataset('dataset_1', data=val_logits)
    h5f2.close()

if __name__ == '__main__':
    # model = cifar10vgg()
    # print('vgg train ok')
    
    cirfar10 = cifar10vgg(train=False)
    model =  cirfar10.return_model()
    extract_feature(model)
