import tensorflow as tf
import tensorflow_addons as tfa


class CNNRoot(tf.keras.layers.Layer):
    """
    [b, h, w, c] -> [b, h/p, w/p, dim]
    """
    def __init__(self, filters, kernel_size, strides, padding, pool_size, pool_strides, **kwargs):
        super(CNNRoot, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.conv = tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters=filters,
                                                   kernel_size=kernel_size,
                                                   strides=strides,
                                                   padding=padding))
        self.gn = tfa.layers.GroupNormalization()
        self.conv_act = tf.keras.layers.Activation('relu')
        self.mp = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding='same')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.gn(x)
        x = self.conv_act(x)
        return self.mp(x)


class ResidualUnitResCNN(tf.keras.layers.Layer):
    """
    [b, h, w, c] -> [b, h, w, c]
    """
    def __init__(self, filters, strides, padding, **kwargs):
        super(ResidualUnitResCNN, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.padding = padding

        self.conv1 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))
        self.conv1_gn = tfa.layers.GroupNormalization()
        self.conv1_act =  tf.keras.layers.Activation('relu')

        self.conv2 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 3, strides, padding='same', use_bias=False))
        self.conv2_gn = tfa.layers.GroupNormalization()
        self.conv2_act = tf.keras.layers.Activation('relu')

        self.conv3 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))
        self.conv3_gn = tfa.layers.GroupNormalization()
        self.conv3_act = tf.keras.layers.ReLU()

        self.conv_res =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1_gn(x)
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.conv2_gn(x)
        x = self.conv2_act(x)
        x = self.conv3(x)
        x = self.conv3_gn(x)
        x = self.conv3_act(x)
        res = self.conv_res(inputs)
        return res + x

class ResidualUnit(tf.keras.layers.Layer):
    """
    [b, h, w, c] -> [b, h, w, c]
    """
    def __init__(self, filters, strides, padding, **kwargs):
        super(ResidualUnit, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.padding = padding

        self.conv1 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))
        self.conv1_gn = tfa.layers.GroupNormalization()
        self.conv1_act = tf.keras.layers.Activation('relu')

        self.conv2 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 3, strides, padding='same', use_bias=False))
        self.conv2_gn = tfa.layers.GroupNormalization()
        self.conv2_act = tf.keras.layers.Activation('relu')

        self.conv3 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))
        self.conv3_gn = tfa.layers.GroupNormalization()
        self.conv3_act = tf.keras.layers.ReLU()

    def call(self, inputs):
        res = inputs
        x = self.conv1(inputs)
        x = self.conv1_gn(x)
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.conv2_gn(x)
        x = self.conv2_act(x)
        x = self.conv3(x)
        x = self.conv3_gn(x)
        x = self.conv3_act(x)
        return res + x


class ResidualUnitShirink(tf.keras.layers.Layer):
    def __init__(self, filters, strides, padding, **kwargs):
        super(ResidualUnitShirink, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.conv1 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides, padding='same', use_bias=False))
        self.conv1_gn = tfa.layers.GroupNormalization()
        self.conv1_act = tf.keras.layers.Activation('relu')
        self.conv2 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 3, strides, padding='same', use_bias=False))
        self.conv2_gn = tfa.layers.GroupNormalization()
        self.conv2_act = tf.keras.layers.Activation('relu')
        self.conv3 =  tfa.layers.WeightNormalization(tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same', use_bias=False))
        self.conv3_gn = tfa.layers.GroupNormalization()
        self.conv3_act = tf.keras.layers.ReLU()
        self.res_mp = tf.keras.layers.MaxPool2D(padding='same')

    def call(self, inputs):
        res = inputs
        x = self.conv1(inputs)
        x = self.conv1_gn(x)
        x = self.conv1_act(x)
        x = self.conv2(x)
        x = self.conv2_gn(x)
        x = self.conv2_act(x)
        x = self.conv3(x)
        x = self.conv3_gn(x)
        x = self.conv3_act(x)
        res_mp = self.res_mp(res)
        return x + res_mp

