"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from .iterator import Iterator
from .utils import array_to_img


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data or tuple.
            If tuple, the second elements is either
            another numpy array or a list of numpy arrays,
            each of which gets passed
            through as an output without any modifications.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataAugmentor`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        sample_weight: Numpy array of sample weights.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataAugmentor.
        dtype: Output dtype into which the generated arrays will be casted before returning
    """

    def __init__(self,
                 x: np.array,
                 y: np.array,
                 image_data_generator,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 sample_weight: np.array = None,
                 data_format: str = 'channels_last',
                 save_to_dir: str = None,
                 save_prefix: str = '',
                 save_format: str = 'png',
                 subset: str = None,
                 dtype: str = 'float32',
                 **kwargs
                 ):
        self.dtype = dtype
        self.seed = image_data_generator.seed

        if (type(x) is tuple) or (type(x) is list):
            if type(x[1]) is not list:
                x_misc = [np.asarray(x[1])]
            else:
                x_misc = [np.asarray(xx) for xx in x[1]]
            x = x[0]
            for xx in x_misc:
                if len(x) != len(xx):
                    raise ValueError(
                        'All of the arrays in `x` '
                        'should have the same length. '
                        'Found a pair with: len(x[0]) = %s, len(x[?]) = %s' %
                        (len(x), len(xx)))
        else:
            x_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError('`x` (images tensor) and `y` (labels) '
                             'should have the same length. '
                             'Found: x.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError('`x` (images tensor) and `sample_weight` '
                             'should have the same length. '
                             'Found: x.shape = %s, sample_weight.shape = %s' %
                             (np.asarray(x).shape, np.asarray(sample_weight).shape))
        if subset is not None:
            if subset not in {'training', 'validation'}:
                raise ValueError('Invalid subset name:', subset,
                                 '; expected "training" or "validation".')
            split_idx = int(len(x) * image_data_generator._validation_split)

            if (y is not None and not
            np.array_equal(np.unique(y[:split_idx]),
                           np.unique(y[split_idx:]))):
                raise ValueError('Training and validation subsets '
                                 'have different number of classes after '
                                 'the split. If your numpy arrays are '
                                 'sorted by the label, you might want '
                                 'to shuffle them.')

            if subset == 'validation':
                x = x[:split_idx]
                x_misc = [np.asarray(xx[:split_idx]) for xx in x_misc]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                x_misc = [np.asarray(xx[split_idx:]) for xx in x_misc]
                if y is not None:
                    y = y[split_idx:]

        self.x = np.asarray(x)
        self.x_misc = x_misc
        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             f'with shape {self.x.shape}'
                             )
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn(f'NumpyArrayIterator is set to use the '
                          f'data format convention `"{data_format}"` '
                          f'(channels on axis {channels_axis}), i.e. '
                          f'expected either 1, 3, or 4 channels on axis '
                          f'{channels_axis}. However, it was passed an array with shape ' 
                          f'{self.x.shape} ({self.x.shape[channels_axis]} channels).'
                          )
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        if sample_weight is not None:
            self.sample_weight = np.asarray(sample_weight)
        else:
            self.sample_weight = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(x.shape[0],
                                                 batch_size,
                                                 shuffle,
                                                 self.seed)

    def _get_batch_of_samples(self, index_array, apply_standardization=True):

        # build batch of image data
        batch_x = np.array([self.x[j] for j in index_array])

        if self.data_format == "channels_first": # swap into "channels_last" data_format if needed
            batch_x = np.array([np.swapaxes(x,0,2) for x in batch_x])

        if self.y is not None:
            batch_y = np.array([self.y[j] for j in index_array])
            data = [
                self.image_data_generator.transform_data(x, y, standardize=apply_standardization)
                for x, y in zip(batch_x, batch_y)
            ]
            batch_x = np.array([d[0] for d in data])
            batch_y = np.array([d[1] for d in data])
        else:
            batch_x = np.array([
                self.image_data_generator.transform_data(x, None, standardize=apply_standardization)[0]
                for x in batch_x
            ])
            batch_y = np.array([])
        if self.data_format == "channels_first": # return the "channels_first" data_format if needed
            batch_x = np.array([np.swapaxes(x,0,2) for x in batch_x])

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # return batch
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        batch_x = batch_x if batch_x_miscs == [] else [batch_x] + batch_x_miscs
        if apply_standardization:
            batch_x = np.asarray(batch_x, dtype=self.dtype)
            batch_y = np.asarray(batch_y, dtype=self.dtype)
        if self.y is None:
            return batch_x
        elif self.sample_weight is None:
            return batch_x, batch_y
        return batch_x, batch_y, self.sample_weight

    def _get_batches_of_transformed_samples(self, index_array):
        return self._get_batch_of_samples(index_array)

    def show_data(self, rows: int = 5, cols: int = 5, apply_standardization: bool = False, **plt_kwargs):
        img_arr = np.random.choice(range(len(self.x)), rows * cols)

        if self.y is None:
            imgs = self._get_batch_of_samples(img_arr, apply_standardization=apply_standardization)
        else:
            imgs = self._get_batch_of_samples(img_arr, apply_standardization=apply_standardization)[0]

        if self.data_format == "channels_first": # swap into the "channels_last" data_format if needed
            imgs = np.array([np.swapaxes(img,0,2) for img in imgs])

        if not 'figsize' in plt_kwargs:
            plt_kwargs['figsize'] = (2 * cols, 2 * rows)

        plt.close('all')
        plt.figure(**plt_kwargs)
        for idx, img in enumerate(imgs):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(img.squeeze())
            plt.axis('off')

        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show()
