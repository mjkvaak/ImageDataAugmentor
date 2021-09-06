## Changes

**2021-09-06**
* Fixed the bug with `ImageDataAugmentor.fit` when custom targets have been specified.

* Changed the signature of `ImageDataAugmentor.preprocess_output` to `ImageDataAugmentor.preprocess_labels`
  (the latter is more descriptive)

* Changed the `DataFrameIterator.class_mode` options `image_target` and `mask_target` to better describing
  `color_target` and `grayscale_target`, respectively.

* Finished the segmentation example `./examples/segmentation-with-flow_from_dataframe.ipynb`

**2020-12-21**

* Added `ImageDataAugmentor.input_augment_mode` that enables selecting augmentations to
inputs

* Logic for `ImageDataAugmentor.input_augment_mode` and `ImageDataAugmentor.label_augment_mode` parameters

* Added an usage example for the aforementioned augment modes in `README.md`


* Fixed some small bugs: unused kwargs will now throw an error, `class_mode==None` returnables fixed


**2020-12-17**

* `ImageDataAugmentor.label_augment_mode` enables targeting augmentations to
labels, e.g if `ImageDataAugmentor.flow_from_dataframe.class_mode` is set to
  `image_target` or `mask_target`

* `Iterator.seed` parameter moved from iterators to `ImageDataAugmentor.seed`: 
  deprecated usage will throw a warning. 
  
* `ImageDataAugmentor.augment_seed` has been removed and is now governed by 
`ImageDataAugmentor.seed` (see above)

* `ImageDataAugmentor.augment_mode` parameter has been removed and the functionality 
  replaced with `ImageDataAugmentor.label_augment_mode` (see above)

* `Iterator.gray` added to color modes (a synonym for `grayscale`)

* `Iterator.show_batch` has been replaced with `Iterator.show_data`

* Added examples to `codebase/examples`

* `Iterator.dtype` now casts the inputs and targets into the desired datatype 
  right before the batch is being returned whereas earlier the datatype casting was
  done in the start of data generation. This will prevent errors caused by augmentations
  not being able to handle the casted datatypes.

* Removed the direct support for `imgaug` since `imgaug` transformations can be called
using `augmentations.imgaug`
