import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()


class DataLoader:
    def __init__(self, data_name: str,
                 data_split: tuple,
                 target_size: int,
                 batch_size: int,
                 normalize: str,
                 patch_img: bool,
                 img_patch_size=16):
        '''
        data_split should follow the order: train, validation, test
        '''
        self.data_name = data_name
        self.target_size = target_size
        self.batch_size = batch_size
        self.normalize = normalize
        self.img_patch_size = img_patch_size
        self.data_split = data_split  # tuple
        self.patch_img = patch_img
        assert self.normalize in ['standard',
                                  'minmax'], f'Requires to be Normalize <standard or minmax> nor {self.normalize}'

    def download_data(self, return_info=False):
        examples, info = tfds.load(self.data_name, as_supervised=True, with_info=True)
        examples_length = len(examples)
        print(f'The Data has {examples_length}')

        if return_info:
            return examples, info
        else:
            return examples

    def split_data(self, examples):
        if len(self.data_split) == 1:
            return examples[self.data_split[0]]

        elif len(self.data_split) == 2:
            return examples[self.data_split[0]], examples[self.data_split[1]]

        elif len(self.data_split) == 3:
            return examples[self.data_split[0]], examples[self.data_split[1]], examples[self.data_split[2]]

    def _resize_img(self, img, label, size):
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (img, label)

    def _standard_normalize(self, img, label):
        img = (img / 127.5) - 1.
        return (img, label)

    def _minmax_normalize(self, img, label):
        img = img / 255.
        return (img, label)

    def _patch_img(self, img, patch_size=16):
        img = tf.expand_dims(img, axis=0)
        img_size = img.shape
        img_height = img_size[1]
        img_width = img_size[2]

        def box_normalize(x):
            return x / img_height

        patchs_result = []
        for h in range(0, img_height, patch_size):
            for w in range(0, img_width, patch_size):
                normalize_box = [box_normalize(h), box_normalize(w), box_normalize(h + patch_size),
                                 box_normalize(w + patch_size)]  # box = [y, x, y1, x1]

                each_patch = tf.image.crop_and_resize(image=img,
                                                      boxes=[normalize_box],
                                                      box_indices=[0],
                                                      crop_size=(patch_size, patch_size))
                patchs_result.append(each_patch)
        return tf.concat(patchs_result, axis=0)

    def _fast_patch_img(self, img, label, patch_size):
        '''
        The official document is unclear
        revisit:
        https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
        '''
        img = tf.image.extract_patches(img,
                                       sizes=[1, patch_size, patch_size, 1],
                                       strides=[1, patch_size, patch_size, 1],
                                       rates=[1, 1, 1, 1],
                                       padding='VALID')
        return (img, label)

    def _tf_patch_image(self, img, label, patch_size=16):
        x = tf.py_function(self._patch_img, inp=[img, patch_size], Tout=tf.float32)
        return (x, label)

    def _transform_img_to_seq(self, img, label, patch_size=16):
        H = int(self.target_size)
        W = int(self.target_size)
        C = 3
        P = int(patch_size)
        N = int((H * W) / (P ** 2))
        img_transform = tf.reshape(img, shape=(N, (P ** 2 * C)))
        return (img_transform, label)

    def _fast_transform_img_to_seq(self, img, label, patch_size=16):
        H = int(self.target_size)
        W = int(self.target_size)
        C = 3
        P = int(patch_size)
        N = int((H * W) / (P ** 2))
        img_transform = tf.reshape(img, shape=(-1, N, (P ** 2 * C)))
        return (img_transform, label)

    def process_data(self, ds_data):
        ds_AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds_data = ds_data.map(lambda x, y: self._resize_img(x, y, size=self.target_size),
                              num_parallel_calls=ds_AUTOTUNE)
        if self.patch_img:
            ds_data = ds_data.map(lambda x, y: self._tf_patch_image(x, y, patch_size=self.img_patch_size),
                                  num_parallel_calls=ds_AUTOTUNE)

        if self.normalize == 'standard':
            ds_data = ds_data.map(lambda x, y: self._standard_normalize(x, y), num_parallel_calls=ds_AUTOTUNE)
        elif self.normalize == 'minmax':
            ds_data = ds_data.map(lambda x, y: self._minmax_normalize(x, y), num_parallel_calls=ds_AUTOTUNE)

        ds_data = ds_data.map(lambda x, y: self._transform_img_to_seq(x, y, patch_size=self.img_patch_size),
                              num_parallel_calls=ds_AUTOTUNE)
        ds_data = ds_data.batch(self.batch_size, drop_remainder=True)
        ds_data = ds_data.cache()
        ds_data = ds_data.prefetch(ds_AUTOTUNE)
        return ds_data

    def batch_process_data(self, ds_data):
        # read >> batch >> map >> cache >> map >> prefetch >> unbatch(opt)
        # Warning: This Method extract the different size of images
        ds_AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds_data = ds_data.batch(self.batch_size, drop_remainder=True)
        ds_data = ds_data.map(lambda x, y: self._resize_img(x, y, size=self.target_size),
                              num_parallel_calls=ds_AUTOTUNE)
        if self.normalize == 'standard':
            ds_data = ds_data.map(lambda x, y: self._standard_normalize(x, y), num_parallel_calls=ds_AUTOTUNE)
        elif self.normalize == 'minmax':
            ds_data = ds_data.map(lambda x, y: self._minmax_normalize(x, y), num_parallel_calls=ds_AUTOTUNE)

        if self.patch_img:
            ds_data = ds_data.map(lambda x, y: self._fast_patch_img(x, y, patch_size=self.img_patch_size),
                                  num_parallel_calls=ds_AUTOTUNE)
        ds_data = ds_data.map(lambda x, y: self._fast_transform_img_to_seq(x, y, patch_size=self.img_patch_size),
                              num_parallel_calls=ds_AUTOTUNE)
        ds_data = ds_data.cache()
        ds_data = ds_data.prefetch(ds_AUTOTUNE)
        return ds_data


if __name__ == '__main__':
    import time

    data_loader = DataLoader('cifar100', ('train', 'test'), 224, 32, 'standard', patch_img=False)
    data = data_loader.download_data()

    tmp_train, tmp_test = data_loader.split_data(data)
    tmp_train_process = data_loader.process_data(tmp_train)
    print(tmp_train_process)

    tmp_train_fast_process = data_loader.batch_process_data(tmp_train)
    print(tmp_train_fast_process)

    _start = time.time()
    for i1, l1 in tmp_train_process.take(1):
        normal_data = i1
    print(time.time() - _start)

    _start = time.time()
    for i2, l2 in tmp_train_fast_process.take(1):
        fast_data = i2
    print(time.time() - _start)