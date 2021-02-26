import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from datahandler.load_data import DataLoader
from utils.load_config import ConfigReader, PrintConfig, SaveConfig
from model.VIT import VisionTransformer
import os



# Initialize Config
config = ConfigReader('config').load_config()

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



vit_model = VisionTransformer(image_size=config['image_height'],
                              patch_size=config['patches'],
                              num_layers=4,
                              num_classes=config['dataset_classes'],
                              model_dim=config['hidden_size'],
                              num_heads=config['num_heads'],
                              feed_forward_units=config['mlp_dim'],
                              dropout_rate=config['dropout_rate'],
                              mlp_size=config['mlp_dim'],
                              first_strides=1,
                              max_pos=1000,
                              res_block=5)



loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)



