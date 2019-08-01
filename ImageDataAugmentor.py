import numpy as np
import threading
import cv2    
from keras.utils import Sequence, to_categorical
import threading
try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None
    
class ImageDataAugmentor(Sequence):
    """Generate batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).
    # Arguments
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
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
        dtype: Dtype to use for the generated arrays.
    """

    def __init__(self,
                 transform = None,
                 rescale=None,
                 preprocess_input=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 dtype='float32'):
                 
        self.transform = transform
        self.rescale = rescale
        self.preprocess_input = preprocess_input
        self.dtype = dtype
        
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
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

    def flow_from_filelist(self, file_list, label_list,
                           target_size=(256, 256), color_mode='rgb',
                           classes=None, class_mode='categorical',
                           batch_size=32, shuffle=True, seed=None,
                           save_to_dir=None,
                           save_prefix='',
                           save_format='png',
                           follow_links=False,
                           subset=None,
                           interpolation=cv2.INTER_NEAREST):
        """Takes the path to a directory & generates batches of augmented data.
        # Arguments
            file_list: List of the target samples.
            label_list: List of the target labels.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "gray", "rbg", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the label_list variable
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
                  which is useful to use with `model.predict_generator()`,
                  `model.evaluate_generator()`, etc.).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
            seed: Optional random seed for shuffling and transformations.
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
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"cv2.INTER_NEAREST"`, `"cv2.INTER_LINEAR"`, `"cv2.INTER_AREA"`, `"cv2.INTER_CUBIC"`
                and `"cv2.INTER_LANCZOS4"`
                By default, `"cv2.INTER_NEAREST"` is used.
        # Returns
            A `FilelistIterator` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return FilelistIterator(
            file_list, label_list, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)
    
    
    def transform_image(self, x):
        if self.transform:
            img = self.transform(x, params)['image']

        return img
    
       
class Iterator(object):
    """Base class for image data iterators.
    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def common_init(self, 
                    image_data_generator,
                    target_size,
                    color_mode,
                    data_format,
                    save_to_dir,
                    save_prefix,
                    save_format,
                    subset,
                    interpolation):
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'bgr', 'rgba', 'gray'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "bgr", "rgba", or "gray".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (4,)
            else:
                self.image_shape = (4,) + self.target_size
        elif self.color_mode == 'rgb' or self.color_mode == 'bgr':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError           
            

class FilelistIterator(Iterator):
    """Iterator capable of reading images from a list of files on disk.
    # Arguments
        file_list: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        label_list: Label of each element in the filelist.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"bgr"`,  `"rgba"`, `"gray"`.
            Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
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
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
        dtype: Dtype to use for generated arrays.
    """

    def __init__(self, file_list, label_list, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation=cv2.INTER_NEAREST,
                 dtype='float32'):
        super(FilelistIterator, self).common_init(image_data_generator,
                                                  target_size,
                                                  color_mode,
                                                  data_format,
                                                  save_to_dir,
                                                  save_prefix,
                                                  save_format,
                                                  subset,
                                                  interpolation)
        self.filenames = file_list
        self.labels = label_list
        
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.dtype = dtype
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        
        # First, count the number of samples and classes.
        self.samples = len(self.filenames)
        
        
        if not classes:
            self.class_indices = dict(zip(np.unique(self.labels),range(len(np.unique(self.labels)))))
            self.num_classes = len(np.unique(self.labels))
        else:
            self.class_indices = dict(zip(np.unique(classes),range(len(np.unique(classes)))))
            self.num_classes = len(classes)
            
        
        self.classes = np.array([self.class_indices[i] for i in self.labels])
       
        print('Found %d images belonging to %d classes.' %
              (self.samples, self.num_classes))

        #print('Self.samples: {}'.format(self.samples))
        #print('Self.classes - len: {0} - {1}'.format(self.classes[0],len(self.classes)))
        #print('Self.filenames - len: {0} - {1}'.format(self.filenames[0],len(self.filenames)))
        #print('Self.class_mode: {}'.format(self.class_mode))
        #print('Self.num_classes: {}'.format(self.num_classes))
        #print('Self.class_indices: {}'.format(self.class_indices))
        
        #Self.samples: 2000
        #Self.classes - len: 0 - 2000
        #Self.filenames - len: cats/cat.0.jpg - 2000
        #Self.class_mode: binary
        #Self.num_classes: 2
        #Self.class_indices: {'cats': 0, 'dogs': 1}
        
        super(FilelistIterator, self).__init__(self.samples,
                                               batch_size,
                                               shuffle,
                                               seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        
        
        ## # build batch of image data
        ## for i, j in enumerate(index_array):
        ##     fname = self.filenames[j]
        ##     img = load_img(fname,
        ##                    color_mode=self.color_mode,
        ##                    target_size=self.target_size,
        ##                    interpolation=self.interpolation)
        ##     x = img_to_array(img, data_format=self.data_format)
        ##     # Pillow images should be closed after `load_img`,
        ##     # but not PIL images.
        ##     if hasattr(img, 'close'):
        ##         img.close()
        ##     x = self.image_data_generator.transform_image(x)
        ##     batch_x[i] = x
        
        # build batch of image data
        batch_x = np.array([load_img(self.filenames(x), 
                                     color_mode=self.color_mode, 
                                     target_size=self.target_size,
                                     interpolation=self.interpolation) for x in index_array])    
        
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
                
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            #batch_y = np.zeros(
            #    (len(batch_x), self.num_classes),
            #    dtype=self.dtype)
            #for i, label in enumerate(self.classes[index_array]):
            #    batch_y[i, label] = 1.
            batch_y = to_categorical(self.classes[index_array], self.num_classes, dtype=self.dtype)
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def save_img(path,
             x,
             data_format='channels_last',
             file_format=None,
             scale=True,
             **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.
    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    img = array_to_img(x, data_format=data_format, scale=scale)
    if img.mode == 'RGBA' and (file_format == 'jpg' or file_format == 'jpeg'):
        warnings.warn('The JPG format does not support '
                      'RGBA images, converting to RGB.')
        img = img.convert('RGB')
    img.save(path, format=file_format, **kwargs)


## def load_img(path, color_mode='rgb', target_size=None,
##              interpolation='nearest'):
##     """Loads an image into PIL format.
##     # Arguments
##         path: Path to image file.
##         color_mode: One of "gray", "rbg", "rgba". Default: "rgb".
##             The desired image format.
##         target_size: Either `None` (default to original size)
##             or tuple of ints `(img_height, img_width)`.
##         interpolation: Interpolation method used to resample the image if the
##             target size is different from that of the loaded image.
##             Supported methods are "nearest", "bilinear", and "bicubic".
##             If PIL version 1.1.3 or newer is installed, "lanczos" is also
##             supported. If PIL version 3.4.0 or newer is installed, "box" and
##             "hamming" are also supported. By default, "nearest" is used.
##     # Returns
##         A PIL Image instance.
##     # Raises
##         ImportError: if PIL is not available.
##         ValueError: if interpolation method is not supported.
##     """
##     if pil_image is None:
##         raise ImportError('Could not import PIL.Image. '
##                           'The use of `array_to_img` requires PIL.')
##     img = pil_image.open(path)
##     if color_mode == 'gray':
##         if img.mode != 'L':
##             img = img.convert('L')
##     elif color_mode == 'rgba':
##         if img.mode != 'RGBA':
##             img = img.convert('RGBA')
##     elif color_mode == 'rgb':
##         if img.mode != 'RGB':
##             img = img.convert('RGB')
##     else:
##         raise ValueError('color_mode must be "gray", "rbg", or "rgba"')
##     if target_size is not None:
##         width_height_tuple = (target_size[1], target_size[0])
##         if img.size != width_height_tuple:
##             if interpolation not in _PIL_INTERPOLATION_METHODS:
##                 raise ValueError(
##                     'Invalid interpolation method {} specified. Supported '
##                     'methods are {}'.format(
##                         interpolation,
##                         ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
##             resample = _PIL_INTERPOLATION_METHODS[interpolation]
##             img = img.resize(width_height_tuple, resample)
##     return img
        
def load_img(fname, color_mode='rgb', target_size=None, interpolation=cv2.INTER_NEAREST):
    if self.color_mode == "rgb":
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    elif self.color_mode == "rgba":
        img = cv2.imread(fname,-1) #Assumes there is an alpha-channel
        if img.shape[-1]!=4
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            
    elif self.color_mode == "gray":
        img = cv2.imread(fname, 0)
        
    else:
        img = cv2.imread(fname)
        
     if target_size is not None:
         width_height_tuple = (target_size[1], target_size[0])
         if img.shape[0:2] != width_height_tuple:
            img = cv2.resize(img, dsize=width_height_tuple, interpolation = interpolation)
    return img
       

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
