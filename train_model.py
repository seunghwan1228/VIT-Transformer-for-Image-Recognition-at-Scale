import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from datahandler.load_data import DataLoader
from utils.load_config import ConfigReader, PrintConfig, SaveConfig
from model.VIT import VisionTransformer
from utils.dynamic_allocation import set_dynamic_allocation
import os
import time
from datetime import datetime
import tqdm



set_dynamic_allocation()
# Initialize Config
config = ConfigReader('config').load_config()

config_dict = config

# Define Data loader -- will use hybrid model, so do not patch input img
data_loader = DataLoader(data_name=config['dataset_name'],
                         data_split=(config['train_prefix'], config['valid_prefix']),
                         target_size=config['image_height'],
                         batch_size=config['batch_size'],
                         normalize=config['normalize'],
                         patch_img=False,
                         img_patch_size=config['patches'])

# Download dataset
data = data_loader.download_data()

# By the datset, split data
# Check Tensorflow dataset catalog
train_set, test_set = data_loader.split_data(data)

# Process dataset
train_data = data_loader.process_data(train_set)
test_data = data_loader.process_data(test_set)

dataset = {'train':train_data,
           'valid':test_data}


model = VisionTransformer(image_size=config['image_height'],
                          patch_size=config['patches'],
                          num_layers=config['num_layers'],
                          num_classes=config['dataset_classes'],
                          model_dim=config['hidden_size'],
                          num_heads=config['num_heads'],
                          feed_forward_units=config['mlp_dim'],
                          dropout_rate=config['dropout_rate'],
                          mlp_size=config['mlp_dim'],
                          first_strides=1,
                          max_pos=1000,
                          res_block=5)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.learning_rate_result = tf.Variable(0.)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr_result = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        self.learning_rate_result.assign(lr_result)
        return lr_result


learning_rate = CustomSchedule(512, warmup_steps=4000)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)


# Metrics related
train_loss_metrics = tf.keras.metrics.Mean(name='Train Loss')
train_acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='Train Accuracy')

test_loss_metrics = tf.keras.metrics.Mean(name='Test Loss')
test_acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='Test Accuracy')

# Checkpoint related
ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
ckpt_save_dir = os.path.join(config_dict['model_save_dir'],
                             f'{config_dict["architecture"]}')

os.makedirs(ckpt_save_dir, exist_ok=True)
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          directory=ckpt_save_dir,
                                          max_to_keep=5,
                                          keep_checkpoint_every_n_hours=3)

# TB Writer related
logdir = config_dict[
             'model_save_dir'] + f'/{config_dict["architecture"]}_{datetime.now().strftime("%Y%m%d")}' + '/logs/train_data/' + datetime.now().strftime(
    "%Y%m%d-%H%M%S")

train_file_writer = tf.summary.create_file_writer(logdir + '/train')
valid_file_writer = tf.summary.create_file_writer(logdir + '/valid')
valid_cm_writer = tf.summary.create_file_writer(logdir + '/cm')


# Train, valid step
@tf.function
def train_step(img, target, sample_weight=None):
    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = loss_obj(target, prediction, sample_weight=sample_weight)
    optimizer.minimize(loss, model.trainable_variables, tape=tape)
    train_loss_metrics(loss)
    train_acc_metrics(target, prediction)


@tf.function
def train_step_mixed_precision(img, target, sample_weight=None):
    with tf.GradientTape() as tape:
        prediction = model(img, training=True)
        loss = loss_obj(target, prediction, sample_weight=sample_weight)
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grad = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_grad)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metrics(loss)
    train_acc_metrics(target, prediction)


@tf.function
def valid_step(img, target, sample_weight=None):
    prediction = model(img, training=False)
    loss = loss_obj(target, prediction, sample_weight=sample_weight)
    test_loss_metrics(loss)
    test_acc_metrics(target, prediction)


def train_model(dataset, epochs):
    train_data = dataset['train']
    valid_data = dataset['valid']

    print('\n\nPreparing Validation Dataset to evaluate\n\n')
    print(f'Training logs are stored at {logdir}')

    iterations = tf.Variable(0, dtype=tf.int64)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print(f'Restart Training from {ckpt_manager.latest_checkpoint}')

    for epoch in tqdm.tqdm(range(epochs)):
        start_time = time.time()
        train_loss_metrics.reset_states()
        test_loss_metrics.reset_states()
        train_acc_metrics.reset_states()
        test_acc_metrics.reset_states()

        print(f'\n\nStart Training {epoch + 1}\n\n')
        for t_n, (img, label) in enumerate(train_data):

            # 21-01-13: Add sample weight
            sample_weights = None

            if config_dict['use_mixed_precision']:
                train_step_mixed_precision(img, label, sample_weight=sample_weights)
            else:
                train_step(img, label, sample_weight=sample_weights)
            iterations.assign_add(1)
            if iterations.numpy() % 10 == 0:
                print(
                    f'Train | Step:: {iterations.numpy()} - Loss:{train_loss_metrics.result()} Acc:{train_acc_metrics.result()}')

            with train_file_writer.as_default():
                tf.summary.scalar(name='train_loss', data=train_loss_metrics.result(), step=iterations)
                tf.summary.scalar(name='train_acc', data=train_acc_metrics.result(), step=iterations)
                tf.summary.scalar(name='Learning_Rate', data=learning_rate.learning_rate_result, step=iterations)
        print(f'Train | {epoch + 1} - Loss:{train_loss_metrics.result()} Acc:{train_acc_metrics.result()}')

        for v_n, (val_img, val_label) in enumerate(valid_data):
            valid_step(val_img, val_label)

            with valid_file_writer.as_default():
                tf.summary.scalar(name='valid_loss', data=test_loss_metrics.result(), step=epoch)
                tf.summary.scalar(name='valid_acc', data=test_acc_metrics.result(), step=epoch)
        print(f'Valid | {epoch + 1} - Loss:{test_loss_metrics.result()} Acc:{test_acc_metrics.result()}')



        print(f'Time Taken:: {epoch + 1}: {time.time() - start_time:.2f}')
        saving_path = ckpt_manager.save()
        print(f'Model Saved at {saving_path}')
    print('Complete Training')


if __name__ == '__main__':
    train_model(dataset=dataset, epochs=config_dict['epochs'])