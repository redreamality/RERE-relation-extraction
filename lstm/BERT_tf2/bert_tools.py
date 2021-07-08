import os, sys, time, math, re
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import *

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr, extend_with_weight_decay
from bert4keras.models import build_transformer_model
from bert4keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal

en_dict_path = r'../tfhub/uncased_L-12_H-768_A-12/vocab.txt'
cn_dict_path = r'../tfhub/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(cn_dict_path, do_lower_case=True)
language = 'cn'

def switch_to_en():
	global tokenizer, language
	tokenizer = Tokenizer(en_dict_path, do_lower_case=True)
	language = 'en'

def convert_sentences(sents, maxlen=256):
	shape = (2, len(sents), maxlen)
	X = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting sentences"):
		tids, segs = tokenizer.encode(sent, maxlen=maxlen)
		X[0,ii,:len(tids)] = tids
		X[1,ii,:len(segs)] = segs
	return [X[0], X[1]]

def convert_tokens(sents, maxlen=256):
	shape = (2, len(sents), maxlen)
	X = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting tokens"):
		tids = tokenizer.tokens_to_ids(sent)
		X[0,ii,:len(tids)] = tids
	return [X[0], X[1]]

def lock_transformer_layers(transformer, layers=-1):
	def _filter(layers, prefix):
		return [x for x in transformer.layers if x.name.startswith(prefix)]
	if hasattr(transformer, 'model'): transformer = transformer.model
	if layers >= 0:
		print('locking', 'Embedding-*')
		for layer in _filter(transformer, 'Embedding-'):
			layer.trainable = False 
	print('locking', 'Transformer-[%d-%d]-*' % (0, layers-1))
	for index in range(layers):
		for layer in _filter(transformer, 'Transformer-%d-' % index):
			layer.trainable = False 

def unlock_transformer_layers(transformer):
	if hasattr(transformer, 'model'): transformer = transformer.model
	for layer in transformer.layers:
		layer.trainable = True

def get_suggested_optimizer(init_lr=5e-5, total_steps=None):
	lr_schedule = {1000:1, 10000:0.01}
	if total_steps is not None: 
		lr_schedule = {total_steps//10:1, total_steps:0.1}
	optimizer = extend_with_weight_decay(Adam)
	optimizer = extend_with_piecewise_linear_lr(optimizer)
	optimizer_params = {
		'learning_rate': init_lr,
		'lr_schedule': lr_schedule,
		'weight_decay_rate': 0.01,
		'exclude_from_weight_decay': ['Norm', 'bias'],
		'bias_correction': False,
	}
	optimizer = optimizer(**optimizer_params)
	return optimizer

def convert_single_setences(sens, maxlen, tokenizer, details=False):
	X = np.zeros((len(sens), maxlen), dtype='int32')
	datas = []
	for i, s in enumerate(sens):
		tokens = tokenizer.tokenize(s)[:maxlen-2]
		if details:
			otokens = restore_token_list(s, tokens)
			datas.append({'id':i, 's':s, 'otokens':otokens})
		tt = ['[CLS]'] + tokens + ['[SEP]']
		tids = tokenizer.convert_tokens_to_ids(tt)
		X[i,:len(tids)] = tids
	if details: return datas, X
	return X

def build_classifier(classes, bert_h5=None):
	if bert_h5 is None:
		bert_h5 = '../tfhub/chinese_roberta_wwm_ext.h5' if language == 'cn' else '../tfhub/bert_uncased.h5'
	bert = load_model(bert_h5)
	output = Lambda(lambda x: x[:,0], name='CLS-token')(bert.output)
	if classes == 2:
		output = Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(stddev=0.02))(output)
	else:
		output = Dense(classes, activation='softmax', kernel_initializer=TruncatedNormal(stddev=0.02))(output)
	model = Model(bert.input, output)
	model.bert_encoder = bert
	return model


## THESE FUNCTIONS ARE TESTED FOR CHS LANGUAGE ONLY
def gen_token_list_inv_pointer(sent, token_list):
	zz = tokenizer.rematch(sent, token_list)
	return [x[0] for x in zz if len(x) > 0]
	sent = sent.lower()
	otiis = []; iis = 0 
	for it, token in enumerate(token_list):
		otoken = token.lstrip('#')
		if token[0] == '[' and token[-1] == ']': otoken = ''
		niis = iis
		while niis <= len(sent):
			if sent[niis:].startswith(otoken): break
			if otoken in '-"' and sent[niis][0] in '—“”': break
			niis += 1
		if niis >= len(sent): niis = iis
		otiis.append(niis)
		iis = niis + max(1, len(otoken))
	for tt, ii in zip(token_list, otiis): print(tt, sent[ii:ii+len(tt.lstrip('#'))])
	for i, iis in enumerate(otiis): 
		assert iis < len(sent)
		otoken = token_list[i].strip('#')
		assert otoken == '[UNK]' or sent[iis:iis+len(otoken)] == otoken
	return otiis

# restore [UNK] tokens to the original tokens
def restore_token_list(sent, token_list):
	if token_list[0] == '[CLS]': token_list = token_list[1:-1]
	invp = gen_token_list_inv_pointer(sent, token_list)
	invp.append(len(sent))
	otokens = [sent[u:v] for u,v in zip(invp, invp[1:])]
	processed = -1
	for ii, tk in enumerate(token_list):
		if tk != '[UNK]': continue
		if ii < processed: continue
		for jj in range(ii+1, len(token_list)):
			if token_list[jj] != '[UNK]': break
		else: jj = len(token_list)
		allseg = sent[invp[ii]:invp[jj]]

		if ii + 1 == jj: continue
		seppts = [0] + [i for i, x in enumerate(allseg) if i > 0 and i+1 < len(allseg) and x == ' ' and allseg[i-1] != ' ']
		if allseg[seppts[-1]:].replace(' ', '') == '': seppts = seppts[:-1]
		seppts.append(len(allseg))
		if len(seppts) == jj - ii + 1:
			for k, (u,v) in enumerate(zip(seppts, seppts[1:])): 
				otokens[ii+k] = allseg[u:v]
		processed = jj + 1
	if invp[0] > 0: otokens[0] = sent[:invp[0]] + otokens[0]
	if ''.join(otokens) != sent:
		raise Exception('restore tokens failed, text and restored:\n%s\n%s' % (sent, ''.join(otokens)))
	return otokens

def gen_word_level_labels(sent, token_list, word_list, pos_list=None):
	otiis = gen_token_list_inv_pointer(sent, token_list)
	wdiis = [];	iis = 0
	for ip, pword in enumerate(word_list):
		niis = iis
		while niis < len(sent):
			if pword == '' or sent[niis:].startswith(pword[0]): break
			niis += 1
		wdiis.append(niis)
		iis = niis + len(pword)
	#for tt, ii in zip(word_list, wdiis): print(tt, sent[ii:ii+len(tt)])

	rlist = [];	ip = 0
	for it, iis in enumerate(otiis):
		while ip + 1 < len(wdiis) and wdiis[ip+1] <= iis: ip += 1
		if iis == wdiis[ip]: rr = 'B'
		elif iis > wdiis[ip]: rr = 'I'
		rr += '-' + pos_list[ip]
		rlist.append(rr)
	#for rr, tt in zip(rlist, token_list): print(rr, tt)
	return rlist

def normalize_sentence(text):
	text = re.sub('[“”]', '"', text)
	text = re.sub('[—]', '-', text)
	text = re.sub('[^\u0000-\u007f\u4e00-\u9fa5\u3001-\u303f\uff00-\uffef·—]', ' \u2800 ', text)
	return text

if __name__ == '__main__':
	switch_to_en()
	sent = 'French is the national language of France where the leaders are FranÃ§ois Hollande and Manuel Valls . Barny cakes , made with sponge cake , can be found in France .'
	tokens = tokenizer.tokenize(sent)
	otokens = restore_token_list(sent, tokens)
	print(tokens)
	print(otokens)
	print('done')