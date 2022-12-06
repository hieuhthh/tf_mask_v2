import os

from utils import *
from layers import *
from model import *
from losses import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

strategy = auto_select_accelerator().scope()
# strategy = tf.device('/cpu:0')

des = path_join(route, 'weights')
mkdir(des)

weight_path = 'best_model_EfficientNetV2S_160_512_73454.h5'
model_weight = path_join(route, weight_path)

encoder_name = 'encoder_v' + str(len(os.listdir(des)) + 1) + '_' + model_weight.split('/')[-1]
encoder_save = path_join(des, encoder_name)

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

n_labels = int(weight_path[:-3].split('_')[-1])
use_cate_int = True

with strategy:
    if emb_pretrain is None:
        base = get_base_model(base_name, input_shape)
        if use_simple_emb:
            emb_model = create_simple_emb_model(base, final_dropout, have_emb_layer, emb_dim)
        else:
            emb_model = create_emb_model(base, final_dropout, have_emb_layer, "embedding",
                                         emb_dim, extract_dim, merge_dim, dilation_rates)
    else:
        emb_model = tf.keras.models.load_model(emb_pretrain, custom_objects={'wBiFPNAdd':wBiFPNAdd, 
                                                                             'PositionEmbedding':PositionEmbedding,
                                                                             'TransformerEncoder':TransformerEncoder})

    model = create_model(input_shape, emb_model, n_labels, use_normdense, use_cate_int, append_norm)
    model.summary()

    emb_name = 'embedding' if emb_pretrain is None else 'sequential'

    if not append_norm:
        losses = {
            'cate_output' : ArcfaceLoss(from_logits=True, 
                                        label_smoothing=arcface_label_smoothing,
                                        margin1=arcface_margin1,
                                        margin2=arcface_margin2,
                                        margin3=arcface_margin3),
            emb_name : SupervisedContrastiveLoss(temperature=sup_con_temperature),
        }
    else:
        losses = {
            'cate_output' : AdaFaceLoss(from_logits=True, 
                                        batch_size=BATCH_SIZE,
                                        label_smoothing=arcface_label_smoothing,
                                        margin=arcface_margin2),
            emb_name : SupervisedContrastiveLoss(temperature=sup_con_temperature),
        }

    loss_weights = {
        'cate_output' : arc_face_weight,
        emb_name : sup_con_weight,
    }

    metrics = {
        'cate_output' : tf.keras.metrics.CategoricalAccuracy()
    }

model.load_weights(model_weight)

with strategy:
    encoder = tf.keras.Sequential([
        model.get_layer('input_1'),
        model.get_layer('embedding'),
    ])
    encoder.summary()
    encoder.save(encoder_save)
    
    print('Save to', encoder_save)

    # encoder = tf.keras.models.load_model(encoder_save)
    # encoder.summary()