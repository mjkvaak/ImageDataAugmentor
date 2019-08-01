# ImageDataAugmentor
Custom image data generator supporting the modern augmentation modules (e.g. imgaug and albumentations) for Keras

 ```
    Example of using `.flow_from_directory(directory)`:
    ```python
    
    transforms = albumentations.Compose([
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
    ```

