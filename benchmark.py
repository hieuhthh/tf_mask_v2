import tensorflow as tf
import os

from layers import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# encoder_save = './weights/encoder_v3_best_model_EfficientNetV2B2_112_512_73501.h5'
# encoder = tf.keras.models.load_model(encoder_save)

encoder_save = 'weights/encoder_v2_best_model_EfficientNetV2S_160_512_73501.h5'
encoder = tf.keras.models.load_model(encoder_save, custom_objects={'wBiFPNAdd':wBiFPNAdd, 
                                                                   'PositionEmbedding':PositionEmbedding,
                                                                   'TransformerEncoder':TransformerEncoder})

encoder.summary()