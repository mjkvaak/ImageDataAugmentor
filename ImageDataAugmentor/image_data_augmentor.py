# Code heavily adapted from:
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/

"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import gc
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy
from scipy import linalg
import cv2
import albumentations
from six.moves import range
from tensorflow.keras.utils import Sequence
from .dataframe_iterator import DataFrameIterator
from .directory_iterator import DirectoryIterator
from .numpy_array_iterator import NumpyArrayIterator


class ImageDataAugmentor(Sequence):
    """Generate batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).
    # Arguments
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocess_input: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function inputs an image and should output a Numpy tensor.
        preprocess_labels: function that will be implied on the labels.
            The function will run after the targets have been augmented.
        augment: augmentations passed as albumentations (sequence of) transformations.
        seed: Optional random seed for shuffling and transformations.
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def __init__(self,
                 featurewise_center: bool = False,
                 samplewise_center: bool = False,
                 featurewise_std_normalization: bool = False,
                 samplewise_std_normalization: bool = False,
                 zca_whitening: bool = False,
                 augment=None,
                 input_augment_mode: str = 'image',
                 label_augment_mode: str = None,
                 seed: int = None,
                 rescale: float = None,
                 preprocess_input: callable = None,
                 preprocess_labels: callable = None,
                 data_format: str = 'channels_last',
                 validation_split: float = 0.0,
                 **kwargs
                 ):
        if kwargs:
            raise TypeError(f"__init__ got  the following unexpected keyword arguments: {', '.join(kwargs.keys())}")
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        if type(augment).__name__ in albumentations.augmentations.transforms.__all__:
            warnings.warn(f"Wrapping single `augment = {type(augment).__name__}` inside `albumentations.Compose()")
            self.augment = albumentations.Compose([augment])
        else:
            self.augment = deepcopy(augment)
        self.input_augment_mode = input_augment_mode
        self.label_augment_mode = label_augment_mode
        if self.augment is not None:
            additional_targets = self.augment.additional_targets
            if self.input_augment_mode not in {'image', 'mask', }.union(additional_targets.keys()):
                raise ValueError(f"Unknown `input_augment_mode='{self.input_augment_mode}'`")
            if (self.label_augment_mode not in {'image', 'mask', 'same_as_input'}.union(additional_targets.keys())
                    and self.label_augment_mode is not None):
                raise ValueError(f"Unknown `label_augment_mode='{self.label_augment_mode}'`")
            if self.label_augment_mode == self.input_augment_mode:
                if "same_as_input" not in additional_targets:
                    warnings.warn(f"Trying to set `label_augment_mode` to `'{input_augment_mode}'`, but this key "
                                  f"has already been preserved for the input image augmentations. "
                                  f"Setting `label_augment_mode='same_as_input'` instead")
                    self.label_augment_mode = "same_as_input"
                    additional_targets.update({self.label_augment_mode: self.input_augment_mode})
                else:
                    raise ValueError(f"Both `input_augment_mode` and `label_augment_mode` cannot be set to "
                                     f"'same_as_input'")
            elif self.label_augment_mode == "same_as_input":
                if "same_as_input" not in additional_targets:
                    warnings.warn(f"Trying to set `label_augment_mode` to `'same_as_input'`, but this key "
                                  f"was not found in augmentations `additional_targets`. "
                                  f"Adding `'{self.label_augment_mode}':'{self.input_augment_mode}'` to `additional_targets`")
                    additional_targets.update({self.label_augment_mode: self.input_augment_mode})
            self.augment = albumentations.Compose(self.augment.transforms.transforms,
                                                  additional_targets=additional_targets
                                                  )
        self.seed = seed
        self.rescale = rescale
        self.preprocess_input = preprocess_input
        self.preprocess_labels = preprocess_labels
        self.total_transformations_done = 0

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2
        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1 or None. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None
        self.principal_components = None

        if zca_whitening:
            if not self.featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if self.featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if self.featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if self.samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataAugmentor specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')

    def flow(self,
             x: np.array,
             y: np.array = None,
             batch_size: int = 32,
             shuffle: bool = True,
             sample_weight=None,
             save_to_dir: str = None,
             save_prefix: str = '',
             save_format: str = 'png',
             subset: str = None,
             dtype: str = 'float32',
             **kwargs
             ):
        """Takes data & label arrays, generates batches of augmented data.
        # Arguments
            x: Input data. Numpy array of rank 4 or a tuple.
                If tuple, the first element
                should contain the images and the second element
                another numpy array or a list of numpy arrays
                that gets passed to the output
                without any modifications.
                Can be used to feed the model miscellaneous data
                along with the images.
                In case of grayscale data, the channels axis of the image array
                should have value 1, in case
                of RGB data, it should have value 3, and in case
                of RGBA data, it should have value 4.
            y: Labels.
            batch_size: Int (default: 32).
            shuffle: Boolean (default: True).
            sample_weight: Sample weights.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataAugmentor`.
        # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the numpy array `x` is returned.
        """
        if kwargs:
            if 'seed' in kwargs:
                warnings.warn('Passing `seed` in `.flow` has been been removed: pass  `seed` '
                              'as parameter in `ImageDataAugmentor(..., seed=...)` instead')
            else:
                raise TypeError(f"__init__ got  the following unexpected keyword arguments: {', '.join(kwargs.keys())}")
        return NumpyArrayIterator(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            dtype=dtype,
            **kwargs
        )

    def flow_from_directory(self,
                            directory: str,
                            target_size: tuple = (256, 256),
                            color_mode: str = 'rgb',
                            classes: list = None,
                            class_mode: str = 'categorical',
                            batch_size: int = 32,
                            shuffle: bool = True,
                            save_to_dir: str = None,
                            save_prefix: str = '',
                            save_format: str = 'png',
                            follow_links: bool = False,
                            subset: str = None,
                            interpolation: str = cv2.INTER_NEAREST,
                            **kwargs
                            ):
        """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale" (or `"gray"`), "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataAugmentor`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        if kwargs:
            if 'seed' in kwargs:
                warnings.warn('Passing `seed` in `.flow_from_directory` has been been removed: pass  `seed` '
                              'as parameter in `ImageDataAugmentor(..., seed=...)` instead')
            else:
                raise TypeError(f"__init__ got  the following unexpected keyword arguments: {', '.join(kwargs.keys())}")
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation
        )

    def flow_from_dataframe(self,
                            dataframe: pd.DataFrame,
                            directory: str = None,
                            x_col: str = "filename",
                            y_col: str = "class",
                            weight_col: str = None,
                            target_size: tuple = (256, 256),
                            color_mode: str = 'rgb',
                            classes: list = None,
                            class_mode: str = 'categorical',
                            batch_size: int = 32,
                            shuffle: bool = True,
                            save_to_dir: str = None,
                            save_prefix: str = '',
                            save_format: str = 'png',
                            subset: str = None,
                            interpolation=cv2.INTER_NEAREST,
                            validate_filenames: bool = True,
                            **kwargs):
        """Takes the dataframe and the path to a directory
         and generates batches of augmented/normalized data.
        **A simple tutorial can be found **[here](
                                    http://bit.ly/keras_flow_from_dataframe).
        # Arguments
            dataframe: Pandas dataframe containing the filepaths relative to
                `directory` (or absolute paths if `directory` is None) of the
                images in a string column. It should include other column/s
                depending on the `class_mode`:
                - if `class_mode` is `"categorical"` (default value) it must
                    include the `y_col` column with the class/es of each image.
                    Values in column can be string/list/tuple if a single class
                    or list/tuple if multiple classes.
                - if `class_mode` is `"binary"` or `"sparse"` it must include
                    the given `y_col` column with class values as strings.
                - if `class_mode` is `"raw"` or `"multi_output"` it should contain
                the columns specified in `y_col`.
                - if `class_mode` is `"input"` or `None` no extra column is needed.
            directory: string, path to the directory to read images from. If `None`,
                data in `x_col` column should be absolute paths.
            x_col: string, column in `dataframe` that contains the filenames (or
                absolute paths if `directory` is `None`).
            y_col: string or list, column/s in `dataframe` that has the target data.
            weight_col: string, column in `dataframe` that contains the sample
                weights. Default: `None`.
            target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to have 1 or 3 color channels.
            classes: optional list of classes (e.g. `['dogs', 'cats']`).
                Default: None. If not provided, the list of classes will be
                automatically inferred from the `y_col`,
                which will map to the label indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: one of "binary", "categorical", "input", "multi_output",
                "raw", sparse" or None. Default: "categorical".
                Mode for yielding the targets:
                - `"binary"`: 1D numpy array of binary labels,
                - `"categorical"`: 2D numpy array of one-hot encoded labels.
                    Supports multi-label output.
                - `"input"`: images identical to input images (mainly used to
                    work with autoencoders),
                - `"multi_output"`: list with the values of the different columns,
                - `"raw"`: numpy array of values in `y_col` column(s),
                - `"sparse"`: 1D numpy array of integer labels,
                - `None`, no targets are returned (the generator will only yield
                    batches of image data, which is useful to use in
                    `model.predict_generator()`).
            batch_size: size of the batches of data (default: 32).
            shuffle: whether to shuffle the data (default: True)
            seed: optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: whether to follow symlinks inside class subdirectories
                (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataAugmentor`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed, `"box"` and
                `"hamming"` are also supported. By default, `"nearest"` is used.
            validate_filenames: Boolean, whether to validate image filenames in
                `x_col`. If `True`, invalid images will be ignored. Disabling this
                option can lead to speed-up in the execution of this function.
                Default: `True`.
        # Returns
            A `DataFrameIterator` yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """
        if kwargs:
            if 'seed' in kwargs:
                warnings.warn('Passing `seed` in `.flow_from_datagrame` has been been removed: pass  `seed` '
                              'as parameter in `ImageDataAugmentor(..., seed=...)` instead')
            else:
                raise TypeError(f"__init__ got  the following unexpected keyword arguments: {', '.join(kwargs.keys())}")

        return DataFrameIterator(
            dataframe,
            directory,
            self,
            x_col=x_col,
            y_col=y_col,
            weight_col=weight_col,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size,
            shuffle=shuffle,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation,
            validate_filenames=validate_filenames
        )

    def transform_data(self,
                       inputs: np.array,
                       targets: np.array = None,
                       standardize: bool = True):
        """
        Transforms an image by first augmenting and then standardizing
        """
        transform_targets = (self.label_augment_mode is not None) and (targets is not None)
        # augmentations
        if self.augment is not None:
            if self.seed:
                random.seed(self.seed + self.total_transformations_done)
            data = {self.input_augment_mode: inputs}
            if self.input_augment_mode != 'image':
                # add dummy 'image' data, since albumentations always expects the `image` field
                data['image'] = np.zeros_like(inputs, dtype=inputs.dtype)
            if transform_targets:
                data[self.label_augment_mode] = targets
            data = self.augment(**data)
            inputs = data[self.input_augment_mode]
            if transform_targets:
                targets = data[self.label_augment_mode]
            self.total_transformations_done += 1
        # standardization
        if standardize:
            inputs, targets = self.standardize(inputs, targets)
        return inputs, targets

    def standardize(self, x, y):
        """Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standarize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:
        standardize(np.copy(x))
        # Arguments
            x: Batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocess_input:
            x = self.preprocess_input(x)
        if self.preprocess_labels:
            y = self.preprocess_labels(y)
        if self.rescale:
            x = np.multiply(x, self.rescale)
        if self.samplewise_center:
            x = x - np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x = np.divide(x, (np.std(x, keepdims=True) + 1e-6))
        if self.featurewise_center:
            if self.mean is not None:
                x = x - self.mean
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_center`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x = x / (self.std + 1e-6)
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`featurewise_std_normalization`, '
                              'but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataAugmentor specifies '
                              '`zca_whitening`, but it hasn\'t '
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x, y

    def fit(self, x,
            augment=False,
            rounds=1,
            ):
        """Fits the data generator to some sample data.
        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.
        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.
        # Arguments
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' +
                self.data_format + '" (channels on axis ' +
                str(self.channel_axis) + '), i.e. expected '
                                         'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                                         'However, it was passed an array with shape ' +
                str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
                ' channels).')

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=x.dtype
            )
            for r in range(rounds):
                for i in range(x.shape[0]):
                    data = {self.input_augment_mode: x[i]}
                    if self.input_augment_mode != 'image':
                        # add dummy 'image' data, since albumentations always expects the `image` field
                        data['image'] = np.zeros_like(x[i], dtype=x.dtype)
                    ax[i + r * x.shape[0]] = self.augment(**data)[self.input_augment_mode]
            x = ax
            del ax, data
            gc.collect()

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x = x - self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x = np.divide(x, (self.std + 1e-6))

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)
