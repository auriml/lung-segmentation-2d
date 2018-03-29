from __future__ import print_function, division

from keras.datasets import mnist
from load_data import loadDataMontgomery, loadDataJSRT, loadDataGeneral
from inference import remove_small_regions, masked, test_benchmark_JSRT
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


        #z = Input(shape=(100,)) #noise as input in classical GAN

        # The generator takes Rx as input and generated masks
        z = Input(self.img_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # Rx images as input => generates masks => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

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

    def _build_generator(self):

        noise_shape = (100,)
        
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)



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


    def _build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_discriminator(self):

        data = Input(shape=self.mask_shape)
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

    def _train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
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


        #(X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # When training involves critic network, for each mini-batch we perform 5 optimization steps
            # on the segmentation network for each optimization step on the critic network.
            indexes = list(range(X_train.shape[0]))
            np.random.shuffle(indexes)
            #indexes = np.array_split(indexes, [batch_size])
            indexes = np.split(indexes, [batch_size])
            indexes = [x for x in indexes if x != []]

            d_loss = None

            k = 5 # a constant, how many times we train gen more than dis
            for iteration, id in enumerate( indexes):
                for X_train_batch, X_masks_train_batch in train_gen.flow(X_train[id], X_masks_train[id], batch_size=batch_size, shuffle=True):
                    if iteration % k ==0:
                        print(str(epoch) + "-training discriminator")
                        # ---------------------
                        #  Train Discriminator
                        # ---------------------

                        # Select half batch of masks
                        masks = X_masks_train_batch[:half_batch]


                        # Generate a half batch of images from noise
                        #noise = np.random.normal(0, 1, (half_batch, 100))
                        #gen_imgs = self.generator.predict(noise)

                        # Generate a half batch of new masks from images
                        # Select a random?? half batch of images or same images that originated the true discriminator masks above? TODO: test both alternatives
                        imgs = X_train_batch[half_batch:]
                        gen_masks = self.generator.predict(imgs)

                        # Train the discriminator
                        d_loss_real = self.discriminator.train_on_batch(masks, np.ones((half_batch, 1)))
                        d_loss_fake = self.discriminator.train_on_batch(gen_masks, np.zeros((half_batch, 1)))
                        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    print(str(epoch) + "-training generator")
                    #noise = np.random.normal(0, 1, (batch_size, 100))
                    imgs = X_train_batch[:batch_size]

                    # The generator wants the discriminator to label the generated samples
                    # as valid (ones)
                    valid_y = np.array([1] * batch_size)

                    # Train the generator
                    g_loss = self.combined.train_on_batch(imgs, valid_y)

                    # Plot the progress
                    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

                    iteration +=1


                    if iteration == len(indexes): #arbitray number of n times data augmentation (change to 1 to not increase training set)
                        break
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.combined.save('gan_combined_model.hdf5')
                self.generator.save('gan_discriminator_model.hdf5')
                self.generator.save('gan_generator_model_post.hdf5')
                test_benchmark_JSRT(model_name='gan_generator_model_post.hdf5' , im_shape= (400,400))

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


        # r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, 100))
        # gen_imgs = self.generator.predict(noise)
        #
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig(os.getcwd() + "/images/mnist_%d.png" % epoch)
        # plt.close()


if __name__ == '__main__':
    gan = GAN()
    #gan.train(epochs=30000, batch_size=32, save_interval=200)
    gan.train(epochs=350, batch_size=10, save_interval=25)






