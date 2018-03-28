from image_gen import ImageDataGenerator
from load_data import loadDataMontgomery, loadDataJSRT
from build_model import build_UNet2D_4L
from gan_model import GAN

import pandas as pd
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import os

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

def load_training_validation_dataJSRT(im_shape = (256,256), n_images=None):
    # Load the dataset
    path = root + '/JSRT/new/'
    y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
    df = pd.DataFrame({'filename':y})
    df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    df = df.sample(frac=1, random_state=23)
    n_images = len(df) if n_images is None else n_images
    n_train = int(n_images*0.8)
    df_train = df[:n_train]
    df_val = df[n_train:n_images]

    # Load training and validation data

    X_train, y_masks_train = loadDataJSRT(df_train, path, im_shape)
    X_val, y_masks_val = loadDataJSRT(df_val, path, im_shape)
    return X_train, y_masks_train, X_val, y_masks_val

def train_UNet(X_train, y_train, X_val, y_val):
    # Build UNet model
    inp_shape = X_train[0].shape
    UNet = build_UNet2D_4L(inp_shape)
    UNet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Visualize model
    plot_model(UNet, 'model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'UNet_model.{epoch:03d}.hdf5'
    print (model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, period=10)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 8
    UNet.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=100,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)
def train_gan_generator(X_train, y_train, X_val, y_val):
    gan = GAN()
    # Visualize model
    plot_model(gan.generator, 'gan_generator_model.png', show_shapes=True)

    ##########################################################################################
    model_file_format = 'gan_generator_model.hdf5'
    print (model_file_format)
    checkpointer = ModelCheckpoint(model_file_format, monitor='val_loss', save_best_only=True)

    train_gen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1.,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   cval=0)

    test_gen = ImageDataGenerator(rescale=1.)

    batch_size = 10
    gan.generator.fit_generator(train_gen.flow(X_train, y_train, batch_size),
                       steps_per_epoch=(X_train.shape[0] + batch_size - 1) // batch_size,
                       epochs=700,
                       callbacks=[checkpointer],
                       validation_data=test_gen.flow(X_val, y_val),
                       validation_steps=(X_val.shape[0] + batch_size - 1) // batch_size)

if __name__ == '__main__':
    # Load training and validation data
    X_train, y_train, X_val, y_val = load_training_validation_dataJSRT(im_shape = (400, 400),n_images=None )
    #train_UNet(X_train, y_train, X_val, y_val)
    train_gan_generator(X_train, y_train, X_val, y_val)

