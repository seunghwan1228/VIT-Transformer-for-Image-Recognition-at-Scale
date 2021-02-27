import tensorflow as tf


def set_dynamic_allocation():
    devices = tf.config.experimental.list_physical_devices('GPU')
    if devices:
        try:
            for gpu in devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(devices), 'Physical Gpu', len(logical_gpus), 'Logical Gpu')
        except RuntimeError as e:
            print(e)
