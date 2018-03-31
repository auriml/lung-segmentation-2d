from __future__ import print_function, division

from keras.datasets import mnist
from load_data import loadDataMontgomery, loadDataJSRT, loadDataGeneral
from inference import remove_small_regions, masked, test_benchmark_JSRT
from train_process_utils import TrainProcessUtil
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
from keras.models import load_model

from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
import random

import matplotlib.pyplot as plt

import sys
import os

import numpy as np

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)


class GAN():
    def __init__(self):
        self.img_rows = 400
        self.img_cols = 400
        self.channels = 1
        #self.mask_channels = 4
        self.mask_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.mask_shape = (self.img_rows, self.img_cols, self.mask_channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        #self.generator = self.build_generator()
        #self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        #Load pretrained generator
        # Load model
        self.generator = load_model('gan_generator_model.hdf5')



        # The generator takes Rx as input and generated masks
        g_inputs = Input(self.img_shape)
        g_masks = self.generator(g_inputs)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated generated masks as input and determines validity
        combined_input = concatenate([g_inputs,g_masks], axis=3)
        valid = self.discriminator(combined_input)

        # The combined model  (stacked generator and discriminator) takes
        # Rx images as input => generates masks => determines validity
        self.combined = Model(g_inputs, outputs=[g_masks,valid])
        combined_loss = ['binary_crossentropy',  'binary_crossentropy']
        loss_weights = [ 1., 0.5]
        metric_dict = [ Dice, 'accuracy']
        self.combined.compile(loss=combined_loss,loss_weights= loss_weights,  optimizer=optimizer,  metrics=metric_dict)


    def residual_block(self, input, filters,kernel_size ):
        shortcut = input
        y = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=(1,1))(input)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        y = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', strides=(1,1))(y)
        y = layers.BatchNormalization()(y)
        shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        y = layers.add([shortcut,y])
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
        return y



    def build_generator(self):

        data = Input(shape=self.img_shape)
        #Resblock 7x7, 8
        y = self.residual_block(data,filters=8, kernel_size=7)

        #Resblock 3x3, 8
        y = self.residual_block(y,filters=8, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)

        #Resblock 3x3, 16
        y = self.residual_block(pool,filters=16, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)


        #Resblock 3x3, 32
        y = self.residual_block(pool,filters=32, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)


        #Resblock 3x3, 64
        y = self.residual_block(pool,filters=64, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)

        #Resblock 1x1, 64
        y = self.residual_block(pool,filters=64, kernel_size=1)

        #Resblock 3x3, 64
        y = self.residual_block(y,filters=64, kernel_size=3)

        #Resblock 1x1, 64
        y = self.residual_block(y,filters=64, kernel_size=1)

        #Resblock 3x3, 64
        y = self.residual_block(y,filters=64, kernel_size=3)

        #Resblock 1x1, 4
        y = self.residual_block(y,filters=4, kernel_size=1)

        #Deconvolution 32 x 32, stride 16, filters 4
        #y = layers.Conv2DTranspose(filters= 4,kernel_size=(32, 32),strides = (16,16), padding="same")(y)

        #Deconvolution 32 x 32, stride 16, filters 1 (output 1 channel if only lungs mask) TODO: use 4 channels for heart + 2 lungs + background
        y = layers.Conv2DTranspose(filters= 1,kernel_size=(32, 32),strides = (16,16), padding="same")(y)
        output = y
        model = Model(data, output)
        model.summary()
        return model




    def build_discriminator(self):

        data = Input(shape=(self.img_rows, self.img_cols,2)) #One channel for lung mask and a second channel for raw image TODO: increase channels for additional organ channel
        #Resblock 7x7, 8
        y = self.residual_block(data,filters=8, kernel_size=7)

        #Resblock 3x3, 8
        y = self.residual_block(y,filters=8, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)

        #Resblock 3x3, 16
        y = self.residual_block(pool,filters=16, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)


        #Resblock 3x3, 32
        y = self.residual_block(pool,filters=32, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)


        #Resblock 3x3, 64
        y = self.residual_block(pool,filters=64, kernel_size=3)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(y)

        #Resblock 1x1, 64
        y = self.residual_block(pool,filters=64, kernel_size=1)

        #Resblock 3x3, 64
        y = self.residual_block(y,filters=64, kernel_size=3)

        #Resblock 1x1, 64
        y = self.residual_block(y,filters=64, kernel_size=1)

        #Resblock 3x3, 64
        y = self.residual_block(y,filters=64, kernel_size=3)

        #Global Average Pooling
        pool = layers.GlobalAveragePooling2D()(y)

        #Fully Connected Layer
        y = layers.Dense(1, activation='sigmoid')(pool)

        output = y
        model = Model(data, output)
        model.summary()
        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        path = root + '/JSRT/new/'
        y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
        df = pd.DataFrame({'filename':y})
        df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)
        # Shuffle rows in dataframe. Random state is set for reproducibility.
        df = df.sample(frac=1, random_state=23)
        n_train = int(len(df)*0.8)
        df_train = df[:n_train]
        df_val = df[n_train:]

        # Load training and validation data
        #im_shape = (256, 256)
        im_shape = (400, 400)
        #X_train, X_masks_train = loadDataJSRT(df_train, path, im_shape, n_images=10) #TODO: n_images is only for testing/debugging
        X_train, X_masks_train = loadDataJSRT(df_train, path, im_shape)

        train_gen = ImageDataGenerator(rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rescale=1.,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       cval=0)

        test_gen = ImageDataGenerator(rescale=1.)



        half_batch = int(batch_size / 2)
        train_process_util = TrainProcessUtil()

        for epoch in range(epochs):
            # When training involves critic network, for each mini-batch we perform 5 optimization steps
            # on the segmentation network for each optimization step on the critic network.
            indexes = list(range(X_train.shape[0]))
            np.random.shuffle(indexes)
            #indexes = np.array_split(indexes, [batch_size])
            indexes = np.split(indexes, [batch_size])
            indexes = [x for x in indexes if x != []]
            print(indexes)

            d_loss = None

            k = 5 # a constant, how many times we train gen more than dis
            for iteration, id in enumerate( indexes):
                for X_train_batch, X_masks_train_batch in train_gen.flow(X_train[id], X_masks_train[id], batch_size=batch_size, shuffle=True):

                    if iteration % k ==0:
                        print(str(epoch) + "-training discriminator")
                        # ---------------------
                        #  Train Discriminator
                        # ---------------------


                        # Generate a  batch of new masks from images
                        gen_masks = self.generator.predict(X_train_batch)

                        # Train the discriminator
                        discriminator_x_true = np.concatenate([X_train_batch, X_masks_train_batch], axis=3)
                        discriminator_x_fake = np.concatenate([X_train_batch, gen_masks], axis=3)
                        discriminator_x = np.concatenate([discriminator_x_true, discriminator_x_fake])
                        discriminator_y = np.ones([2 * batch_size, 1])
                        discriminator_y[batch_size:, :] = 0

                        d_loss = self.discriminator.train_on_batch(discriminator_x, discriminator_y)
                        discriminator_log_message = "%d: [Discriminator model loss: %f, accuracy: %f]" % (iteration, d_loss[0], d_loss[1])
                        print(discriminator_log_message)


                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    print(str(epoch) + "-training adversarial")

                    # The generator wants the discriminator to label the generated samples
                    # as valid (ones)
                    valid_y = np.array([1] * batch_size)
                    label_dict = [ X_masks_train_batch, valid_y]

                    # Train the generator
                    g_loss = self.combined.train_on_batch(X_train_batch, label_dict)

                    # Plot the progress
                    message = "%d [D loss: %f, acc.: %.2f%%] " % (epoch, d_loss[0], 100*d_loss[1])
                    print (message)
                    f = open('test_benchmark_JSRT', 'a')
                    f.write(message + '\n')
                    f.close()

                    metric_names = self.combined.metrics_names
                    log_message = "%d:" % iteration
                    for metric_index in range(len(metric_names)):
                        train_process_util.update_metrics_dict(metric_names[metric_index], g_loss[metric_index])
                        message = "%s: %f   " % ( metric_names[metric_index], g_loss[metric_index])
                        log_message += message
                    print(log_message)
                    f = open('test_benchmark_JSRT', 'a')
                    f.write( log_message + '\n')
                    f.close()

                    if iteration == 1: #arbitray number of n times data augmentation (change to 1 to not increase training set)
                        break

                    iteration +=1
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.combined.save('gan_combined_model.hdf5')
                self.generator.save('gan_discriminator_model.hdf5')
                self.generator.save('gan_generator_model_post.hdf5')
                f = open('test_benchmark_JSRT', 'a')
                f.write("Epoch " + str(epoch) + '\n' )
                f.close()
                test_benchmark_JSRT(model_name='gan_generator_model_post.hdf5' , im_shape= (400,400))

            #train_process_util.plot_metrics() //TODO fix

    def save_imgs(self, epoch):
        path = root + '/Rx-thorax-automatic-captioning/image_dir_processed/'
        y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
        df = pd.DataFrame({'filename':y})
        df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)

        # Load test data
        im_shape = (400, 400)
        X, y = loadDataGeneral(df, path, im_shape, n_images=10)

        n_test = X.shape[0]
        inp_shape = X[0].shape



        # For inference standard keras ImageGenerator is used.
        test_gen = ImageDataGenerator(rescale=1.)


        i = 0

        for xx in X:
            img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
            xx = np.expand_dims(xx, 0)

            gen_masks = self.generator.predict(xx)
            pred = gen_masks[..., 0].reshape(inp_shape[:2])
            pr = pred > 0.5

            # Remove regions smaller than 2% of the image
            pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
            io.imsave(os.getcwd() + '/gan_results/{}'.format(str(epoch) + '-' + df.iloc[i][0]), masked(img,  pr, 1))
            i += 1



def Dice(y_true, y_pred):
    smooth = 1.
    y_true_f = K.sigmoid(K.flatten(y_true))
    y_pred_f = K.sigmoid(K.flatten(y_pred))

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -Dice(y_true, y_pred)

if __name__ == '__main__':
    gan = GAN()
    #gan.train(epochs=30000, batch_size=32, save_interval=200)
    gan.train(epochs=1000, batch_size=20, save_interval=25)
    #gan.train(epochs=100, batch_size=4, save_interval=25)






