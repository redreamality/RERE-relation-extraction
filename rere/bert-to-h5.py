import tensorflow as tf
sys.path.append('./BERT_tf2')
from bert4keras.models import build_transformer_model

bert_path = './tfhub/uncased_L-12_H-768_A-12'
bert = build_transformer_model(bert_path, return_keras_model=False) 
bert.model.save('./tfhub/bert_uncased.h5')

