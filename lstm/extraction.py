import os, sys, time, re, utils, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append('./BERT_tf2')
import bert_tools as bt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from bert4keras.backend import keras, K
from collections import defaultdict
import numpy as np
from tqdm import tqdm

datan=sys.argv[1]
dname = f'{datan}_lstm'
datadir = f'../data/{datan}'
trains = utils.LoadJsons(os.path.join(datadir, 'new_train.json'))
valids = utils.LoadJsons(os.path.join(datadir, 'new_valid.json'))
tests = utils.LoadJsons(os.path.join(datadir, 'new_test.json'))

if not os.path.isdir('model/'+dname): os.makedirs('model/'+dname)

def wdir(x): return 'model/'+dname+'/'+x

rels = utils.TokenList(wdir('rels.txt'), 1, trains, lambda z:[x['label'] for x in z['relationMentions']])
print('rels:', rels.get_num())

bt.switch_to_en()

maxlen = 128

def dgcnn_block(x, dim, dila=1):
    y1 = Conv1D(dim, 3, padding='same', dilation_rate=dila)(x)
    y2 = Conv1D(dim, 3, padding='same', dilation_rate=dila, activation='sigmoid')(x)
    yy = multiply([y1, y2])
    if yy.shape[-1] == x.shape[-1]: yy = add([yy, x])
    return yy
    
def neg_log_mean_loss(y_true, y_pred):
    eps = 1e-6
    pos = - K.sum(y_true * K.log(y_pred+eps), 1) / K.maximum(eps, K.sum(y_true, 1))
    neg = K.sum((1-y_true) * y_pred, 1) / K.maximum(eps, K.sum(1-y_true, 1))
    neg = - K.log(1 - neg + eps)
    return K.mean(pos + neg * 15)

def FindValuePos(sent, value):
    ret = [];  
    value = value.replace(' ', '').lower()
    if value == '': return ret
    ss = [x.replace(' ', '').lower() for x in sent]
    for k, v in enumerate(ss):
        if not value.startswith(v): continue
        vi = 0
        for j in range(k, len(ss)):
            if value[vi:].startswith(ss[j]):
                vi += len(ss[j])
                if vi == len(value):
                    ret.append( (k, j+1) )
            else: break
    return ret

def GetTopSpans(tokens, rr, K=40):
    cands = defaultdict(float)
    start_indexes = sorted(enumerate(rr[:,0]), key=lambda x:-x[1])[:K]
    end_indexes = sorted(enumerate(rr[:,1]), key=lambda x:-x[1])[:K]
    for start_index, start_score in start_indexes:
        if start_score < 0.1: continue
        if start_index >= len(tokens): continue
        for end_index, end_score in end_indexes:
            if end_score < 0.1: continue
            if end_index >= len(tokens): continue
            if end_index < start_index: continue
            length = end_index - start_index + 1
            if length > 40: continue
            ans = ''.join(tokens[start_index:end_index+1]).strip()
            if '》' in ans: continue
            if '、' in ans and len(ans.split('、')) > 2 and '，' not in ans and ',' not in ans:
                aas = ans.split('、')
                for aa in aas: cands[aa.strip()] += start_score * end_score / len(aas)
                continue
            cands[ans] += start_score * end_score

    cand_list = sorted(cands.items(), key=lambda x:len(x[0]))
    removes = set()
    contains = {}
    for i, (x, y) in enumerate(cand_list):
        for j, (xx, yy) in enumerate(cand_list[:i]):
            if xx in x and len(xx) < len(x):
                contains.setdefault(x, []).append(xx)

    for i, (x, y) in enumerate(cand_list):
        sump = sum(cands[z] for z in contains.get(x, []) if z not in removes)
        suml = sum(len(z) for z in contains.get(x, []) if z not in removes)
        if suml > 0: sump = sump * min(1, len(x) / suml)
        if sump > y: removes.add(x)
        else:
            for z in contains.get(x, []): removes.add(z)

    ret = [x for x in cand_list if x[0] not in removes]
    ret.sort(key=lambda x:-x[1])
    return ret[:K]

def GenTriple(p, x, y):
    return {'label':p, 'em1Text':x, 'em2Text':y}


class RCModel:
    def __init__(self):
        inp_words = Input((None,), dtype='int32')
        inp_seg = Input((None,), dtype='int32')
        xx = Embedding(bt.tokenizer._vocab_size, 256, mask_zero=True)(inp_words)
        xx = Bidirectional(LSTM(256, return_sequences=True))(xx)
        xx = Bidirectional(LSTM(256, return_sequences=True))(xx)
        xx = GlobalAveragePooling1D()(xx)
        #xx = Lambda(lambda x:x[:,0])(xx)
        pos = Dense(rels.get_num(), activation='sigmoid')(xx)
        self.model = tf.keras.models.Model(inputs=[inp_words, inp_seg], outputs=pos)
        #bt.lock_transformer_layers(self.bert, 8)
        self.model_ready = False

    def gen_golden_y(self, datas):
        for dd in datas:
            dd['rc_obj'] = list(set(x['label'] for x in dd.get('relationMentions', [])))
        
    def make_model_data(self, datas):
        self.gen_golden_y(datas)
        for dd in tqdm(datas, desc='tokenize'):
            s = dd['sentText']
            tokens = bt.tokenizer.tokenize(s, maxlen=maxlen)
            dd['tokens'] = tokens
        N = len(datas)
        X = [np.zeros((N, maxlen), dtype='int32'), np.zeros((N, maxlen), dtype='int32')]
        Y = np.zeros((N, rels.get_num()))
        for i, dd in enumerate(tqdm(datas, desc='gen XY', total=N)):
            tokens = dd['tokens']
            X[0][i][:len(tokens)] = bt.tokenizer.tokens_to_ids(tokens)
            for x in dd['rc_obj']: Y[i][rels.get_id(x)] = 1
        return X, Y

    def load_model(self):
        self.model.load_weights(wdir('rc.h5'))
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model_ready = True

    def train(self, datas, batch_size=32, epochs=10):
        self.X, self.Y = self.make_model_data(datas)
        #self.optimizer = bt.get_suggested_optimizer(5e-5, len(datas) * epochs // batch_size)
        self.optimizer = RMSprop(5e-5)
        self.model.compile(self.optimizer, 'binary_crossentropy', metrics=['accuracy'])
        self.cb_mcb = ModelCheckpoint(wdir('rc.h5'), save_weights_only=True, verbose=1)
        self.model.fit(self.X, self.Y, batch_size, epochs=epochs, shuffle=True, 
                    validation_split=0.01, callbacks=[self.cb_mcb])
        self.model_ready = True
                
    def get_output(self, datas, pred, threshold=0.5):
        for dd, pp in zip(datas, pred):
            dd['rc_pred'] = list(rels.get_token(i) for i, sc in enumerate(pp) if sc > threshold)

    def evaluate(self, datas):
        ccnt, gcnt, ecnt = 0, 0, 0
        for dd in datas:
            plabels = set(dd['rc_pred'])
            ecnt += len(plabels)
            gcnt += len(set(dd['rc_obj']))
            ccnt += len(plabels & set(dd['rc_obj']))
        return utils.CalcF1(ccnt, ecnt, gcnt)

    def predict(self, datas, threshold=0.5, ofile=None):
        if not self.model_ready: self.load_model()
        self.vX, self.vY = self.make_model_data(datas)
        pred = self.model.predict(self.vX, batch_size=64, verbose=1)
        self.get_output(datas, pred, threshold)
        f1str = self.evaluate(datas)
        if ofile is not None:
            utils.SaveList(map(lambda x:json.dumps(x, ensure_ascii=False), datas), wdir(ofile))
        print(f1str)
        return f1str

class EEModel:
    def __init__(self):
        inp_words = Input((None,), dtype='int32')
        inp_seg = Input((None,), dtype='int32')
        xx = Embedding(bt.tokenizer._vocab_size, 256, mask_zero=True)(inp_words)
        xx = Bidirectional(LSTM(256, return_sequences=True))(xx)
        xx = Bidirectional(LSTM(256, return_sequences=True))(xx)
        pos = Dense(4, activation='sigmoid')(xx)
        self.model = tf.keras.models.Model(inputs=[inp_words, inp_seg], outputs=pos)
        self.model_ready = False
        
    def make_model_data(self, datas):
        if 'tokens' not in datas[0]:
            for dd in tqdm(datas, desc='tokenize'):
                s = dd['sentText']
                tokens = bt.tokenizer.tokenize(s, maxlen=maxlen)
                dd['tokens'] = tokens
        N = 0
        for dd in tqdm(datas, desc='matching'):
            otokens = bt.restore_token_list(dd['sentText'], dd['tokens'])
            dd['otokens'] = otokens
            if '' in otokens:
                print(dd['sentText'])
                print(dd['tokens'])
                print(otokens)
                # assert '' not in otokens
            ys = {}
            if 'rc_pred' in dd:
                plist = dd['rc_pred']
            else:
                for x in dd.get('relationMentions', []):
                    ys.setdefault(x['label'], []).append( (x['em1Text'], x['em2Text']) )
                plist = sorted(ys.keys())
            yys = []
            for pp in plist:
                spos, opos = [], []
                for s, o in ys.get(pp, []):
                    ss, oo = FindValuePos(otokens, s), FindValuePos(otokens, o)
                    if len(ss) == 0 and len(oo) == 0: continue
                    spos.extend(ss)
                    opos.extend(oo)
                yys.append( {'pp':pp, 'spos':spos, 'opos':opos} )
            dd['ee_obj'] = yys
            N += len(yys)
        X = [np.zeros((N, maxlen), dtype='int32'), np.zeros((N, maxlen), dtype='int8')]
        Y = np.zeros((N, maxlen, 4), dtype='int8')
        ii = 0
        for dd in tqdm(datas, desc='gen EE XY'):
            tokens = dd['tokens']
            for item in dd['ee_obj']:
                pp, spos, opos = item['pp'], item['spos'], item['opos']
                first = bt.tokenizer.tokenize(pp)
                offset = len(first)
                item['offset'] = offset
                tts = (first + tokens[1:])[:maxlen]
                X[0][ii][:len(tts)] = bt.tokenizer.tokens_to_ids(tts)
                X[1][ii][offset:offset+len(tokens)-1] = 1
                for u, v in spos:
                    try:
                        Y[ii][offset+u,0] = 1
                        Y[ii][offset+v-1,1] = 1
                    except: pass
                for u, v in opos:
                    try:
                        Y[ii][offset+u,2] = 1
                        Y[ii][offset+v-1,3] = 1
                    except: pass
                ii += 1
        return X, Y

    def train(self, datas, batch_size=32, epochs=10):
        self.X, self.Y = self.make_model_data(datas)
        #self.optimizer = bt.get_suggested_optimizer(5e-5, len(datas) * epochs // batch_size)
        self.optimizer = RMSprop(5e-5)
        self.model.compile(self.optimizer, 'binary_crossentropy', metrics=['accuracy'])
        self.cb_mcb = ModelCheckpoint(wdir('ee.h5'), save_weights_only=True, verbose=1)
        self.model.fit(self.X, self.Y, batch_size, epochs=epochs, shuffle=True, 
                    validation_split=0.01, callbacks=[self.cb_mcb])
        self.model_ready = True

    def load_model(self):
        self.model.load_weights(wdir('ee.h5'))
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model_ready = True

    def get_output(self, datas, pred, threshold=0.5):
        ii = 0
        for dd in datas:
            rtriples = []
            for item in dd['ee_obj']:
                predicate, offset = item['pp'], item['offset']
                rr = pred[ii]; ii += 1
                subs = GetTopSpans(dd['otokens'], rr[offset:,:2])
                objs = GetTopSpans(dd['otokens'], rr[offset:,2:])
            
                vv1 = [x for x,y in subs if y >= 0.1]
                vv2 = [x for x,y in objs if y >= 0.1]

                subv = {x:y for x,y in subs}
                objv = {x:y for x,y in objs}

                #mats = None
                #if len(vv1) * len(vv2) >= 4:
                #    sent = ''.join(data[2])
                #    mats = set(Match(sent, vv1, vv2))

                for sv1, sv2 in [(sv1, sv2) for sv1 in vv1 for sv2 in vv2] :
                    if sv1 == sv2: continue
                    score = min(subv[sv1], objv[sv2])
                    #if mats is not None and (sv1, sv2) not in mats: score -= 0.5
                    if score < threshold: continue
                    rtriples.append( GenTriple(predicate, sv1, sv2) )

            dd['ee_pred'] = rtriples
            # assert '' not in dd['otokens']

    def evaluate(self, datas):
        ccnt, gcnt, ecnt = 0, 0, 0
        for dd in datas:
            golden = set();  predict = set()
            for x in dd['relationMentions']:
                ss = '|'.join([x[nn] for nn in ['label', 'em1Text', 'em2Text']])
                golden.add(ss)
            for x in dd['ee_pred']:
                ss = '|'.join([x[nn] for nn in ['label', 'em1Text', 'em2Text']])
                predict.add(ss)
            ecnt += len(predict)
            gcnt += len(golden)
            ccnt += len(predict & golden)
        return utils.CalcF1(ccnt, ecnt, gcnt)

    def predict(self, datas, threshold=0.5, ofile=None):
        if not self.model_ready: self.load_model()
        self.vX, self.vY = self.make_model_data(datas)
        pred = self.model.predict(self.vX, batch_size=64, verbose=1)
        self.get_output(datas, pred, threshold=threshold)
        if ofile is not None:
            utils.SaveList(map(lambda x:json.dumps(x, ensure_ascii=False), datas), wdir(ofile))
        f1str = self.evaluate(datas)
        print(f1str)
        return f1str
            

if __name__ == '__main__':
    rc = RCModel()
    if 'trainrc' in sys.argv:
        rc.train(trains, batch_size=64, epochs=10)
    if not 'eeonly' in sys.argv:
        rc.predict(tests, threshold=0.4, ofile='valid_rc.json')
        tests = utils.LoadJsons(wdir('valid_rc.json')) 
    ee = EEModel()
    if 'trainee' in sys.argv:
        ee.train(trains, batch_size=32, epochs=10)
    ee.predict(tests, threshold=0.2, ofile='valid_ee.json')
    print('done')