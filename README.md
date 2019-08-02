# ImageDataAugmentor
ImageDataAugmentor is a custom image data generator for Keras supporting the use of modern augmentation modules (e.g. imgaug and albumentations).

**NOTICE!**
The code is heavily adapted from: https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/

The usage is analogous to Keras' `ImageDataGenerator` with the exception that the image transformations will be generated with an external augmentations module. 

To learn more about:
* ImageDataGenerator, see: https://keras.io/preprocessing/image/
* albumentations, see: https://github.com/albu/albumentations
* imgaug, see: https://github.com/aleju/imgaug

For similar projects, see:
* https://github.com/davidfreire/Augmentation_project <- a generator that accepts both external and Keras internal augmentations

> Example of using `.flow_from_directory(directory)` with `albumentations`:

    from ImageDataAugmentor.image_data_augmentor import *
    import albumentations
    
    ...
        
    transform = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            albumentations.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1)
        ],p=1),
        albumentations.GaussianBlur(p=0.05),
        albumentations.HueSaturationValue(p=0.5),
        albumentations.RGBShift(p=0.5),
    ])
    
    train_datagen = ImageDataAugmentor(
            rescale=1./255,
            transform = transform,
            preprocess_input=None)
            
    test_datagen = ImageDataAugmentor(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')
            
    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')
            
    model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=50,
            validation_data=validation_generator,
            validation_steps=len(validation_generator))


> Example of using `.flow(x, y)` with `imgaug`:
    
    from ImageDataAugmentor.image_data_augmentor import *
    from imgaug import augmenters as iaa
    import imgaug as ia
    
    ...
    
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-5, 5), # shear by -5 to +5 degrees
            mode=ia.ALL # use any of scikit-image's warping modes
        )
        )],
        random_order=True)    
    transform = seq.augment_image
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        transform = transform)
    
    # compute quantities required for featurewise normalization
    datagen.fit(x_train)
    
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs)
    


CITE (BibTex):

@misc{mjkvaak_aug,
author = {Tukiainen, M.},
title = {ImageDataAugmentor},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/mjkvaak/ImageDataAugmentor/}}
}
