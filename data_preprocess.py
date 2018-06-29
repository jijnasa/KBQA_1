''' Training Data:
    KBQA_RE_data/webqsp_relations/relations.txt
    KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt
    KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt
'''
import sys
import os
import time
import gensim
import numpy as np

import torch
from torch.autograd import Variable
import math
class DataManager:
    def __init__(self):
        self.train_data_path = 'KBQA_RE_data/webqsp_relations/WebQSP.RE.train.with_boundary.withpool.dlnlp.txt'
        self.test_data_path = 'KBQA_RE_data/webqsp_relations/WebQSP.RE.test.with_boundary.withpool.dlnlp.txt'
#        self.train_data_path = '1hop_2hop_data/WebQSP.1hop'
#        self.test_data_path =  '1hop_2hop_data/WebQSP.2hop'
        self.word_embedding_path = 'KBQA_RE_word_emb_300d.txt'
        self.rela_embedding_path = 'KBQA_RE_rela_emb_300d.txt'
        #self.word_embedding_path = 'KBQA_RE_word_emb_300d_last.txt'
        #self.rela_embedding_path = 'KBQA_RE_rela_emb_300d_last.txt'
        self.emb_dim = 300
        self.relations_map = {}
        self.word_dic = {}
        self.word_embedding = []
        self.rela_dic = {}
        self.rela_embedding = []

        self.relations_map = self.load_relations_map('KBQA_RE_data/webqsp_relations/relations.txt')
        #print('Original training questions: 3116')
        #print('Original testing questions: 1649')
        train_data = self.gen_train_data(self.train_data_path) 
        test_data = self.gen_train_data(self.test_data_path) 
        print('Filter out questions without negative training samples.')
        print(f'Train data length:{len(train_data)}')
        print(f'Test data length:{len(test_data)}')

        if not os.path.isfile(self.word_embedding_path):
            print(self.word_embedding_path, 'not exist!')
            self.save_embeddings(train_data+test_data)
            print()
        self.word_dic, self.word_embedding = self.load_embeddings(self.word_embedding_path)
        self.rela_dic, self.rela_embedding = self.load_embeddings(self.rela_embedding_path)
        token_train_data = self.tokenize_train_data(train_data)
        token_test_data = self.tokenize_train_data(test_data)
        print(f'Tokened train data length:{len(token_train_data)}')
        print(f'Tokened test data length:{len(token_test_data)}')
        self.maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = self.find_maxlength(token_train_data+token_test_data)
        self.maxlen_r = max(maxlen_pos_r, maxlen_neg_r)
        self.maxlen_w = max(maxlen_pos_w, maxlen_neg_w)
        print(f'maxlen_q:{self.maxlen_q}, maxlen_r:{self.maxlen_r}, maxlen_w:{self.maxlen_w}')
        #print(f'maxlen_pos_r:{maxlen_pos_r}, maxlen_neg_r:{maxlen_neg_r}')
        #print(f'maxlen_pos_w:{maxlen_pos_w}, maxlen_neg_w:{maxlen_neg_w}')
        self.token_train_data = self.pad_train_data(token_train_data, self.maxlen_q, self.maxlen_r, self.maxlen_w, self.maxlen_r, self.maxlen_w)
        self.token_test_data = self.pad_train_data(token_test_data, self.maxlen_q, self.maxlen_r, self.maxlen_w, self.maxlen_r, self.maxlen_w)
#        self.check_input_data(test_data, self.token_test_data)

    def check_input_data(self, origin_data, token_data):
        print()
        print('Check token result')
        print('# objs in the 1st question of test data:', len(test_data[0]), len(self.token_test_data[0]))
        print('1st obj in the 1st question of test data:')
        print('obj: (question, pos_relas, pos_words, neg_relas, neg_words)')
        print(origin_data[0][0])
        print(token_data[0][0])
        print(self.idx2word(token_data[0][0][0])) 
        print(self.idx2word(token_data[0][0][1], 'relation'))
        print(self.idx2word(token_data[0][0][2]))
        print(self.idx2word(token_data[0][0][3], 'relation'))
        print(self.idx2word(token_data[0][0][4]))
        print()
    
    def idx2word(self, id_sentence, id_type='word'):
        if id_type == 'relation':
            dic = {idx:word for word, idx in self.rela_dic.items()}
        else:
            dic = {idx:word for word, idx in self.word_dic.items()}
        word_sentence = []
        for idx in id_sentence:
            if idx == 0:
                continue
            word_sentence.append(dic[idx])
        return ' '.join(word_sentence)

    def gen_train_data(self, path):
        ''' Return training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        '''
        data_list = []
        #pos_gt_1_counter = 0
        print('Load', path)
        #start = time.time()
        with open(path) as infile:
            for line in infile: 
                q_list = []
                tokens = line.strip().split('\t')
                pos_relations = tokens[0]
                neg_relations = tokens[1]
                question = tokens[2].replace('$ARG1','').replace('$ARG2','').strip().split(' ')
                #if len(pos_relations.split(' ')) != 1:
                #    pos_gt_1_counter += 1
                #for pos_id in pos_relations.split(' '):
                pos_id = pos_relations.split(' ')[0]
                pos_relas, pos_words = self.split_relation(pos_id)
                for neg_id in neg_relations.split(' '):
                    # skip blank relation (relation_id 1797 = '')
                    if neg_id == '1797':
                        continue
                    neg_relas, neg_words = self.split_relation(neg_id)
                    q_list.append((question, pos_relas, pos_words, neg_relas, neg_words))
                    #print(q_list)
                    #sys.exit()
                if len(q_list) > 0:
                    data_list.append(q_list)
        #print(f'Time elapsed:{time.time()-start:.2f}')
        #print('pos_gt_1_counter', pos_gt_1_counter)
        return data_list

    def split_relation(self, relation_id):
        '''Return relation_token_list and relation_token_name_list
        '''
        rela_list = []
        word_list = []
        relation_names = self.relations_map[int(relation_id)]
        for relation_name in relation_names.split('..'):
            #last_name = relation_name.split('.')[-1]
            #rela_list.append(last_name)
            #for word in last_name.split('_'):
            #    word_list.append(word)
            for relation_token in relation_name.split('.'):
                rela_list.append(relation_token)
                for word in relation_token.split('_'):
                    word_list.append(word)
        return rela_list, word_list

    def find_unique(self, data):
        words = set()
        relas = set()
        #start = time.time()
        for idx, q_data in enumerate(data, 1):
            print('\r# of questions', idx, end='')
            try:
                words |= set(q_data[0][0])
                for data_obj in q_data:
                    relas |= set(data_obj[1]) | set(data_obj[3])
                    words |= set(data_obj[2]) | set(data_obj[4])
            except:
                print(idx, q_data)
        print()
        #relas.remove('')
        if '' in words:
            words.remove('')
        print(f'There are {len(relas)} unique relations and {len(words)} unique words.')
        #print(f'Time elapsed:{time.time()-start:.2f}')
        return relas, words

    def load_word_embedding_from_gensim(self, input_path):
        print('Load pretrain word embedding from', input_path)
        #start = time.time()
        model = gensim.models.Word2Vec.load_word2vec_format(input_path, binary=True)
        #print(f'Time elapsed:{time.time()-start:.2f}') 
        return model

    def load_embeddings(self, path):
        vocab_dic = {}
        embedding = []
        print('Load embedding from', path)
        with open(path) as infile:
            for line in infile:
                tokens = line.strip().split()
                vocab_dic[tokens[0]] = len(vocab_dic)
                embedding.append([float(x) for x in tokens[1:]]) 
        embedding = np.array(embedding)
        # Check embedding correctness
        if embedding.shape[0] != len(vocab_dic):
            print(f'Load embedding error: embedding.shape[0]={embedding_shape[0]} vocab_size={len(vocab_size)}')
            sys.exit()
        if embedding.shape[1] != self.emb_dim:
            print(f'Load embedding error: embedding.shape[1]={embedding_shape[1]} emb_dim={self.emb_dim}')
            sys.exit()
        return vocab_dic, embedding

    def save_embeddings(self, data):
        rela_set, word_set = self.find_unique(data)

        # Load 300 dim pretrained word2vec embeddings trained on GoogleNews. 
        # To be more efficient, only load words contains in training/testing data.
        exception_counter = 0
        input_w2v_path = '/corpus/wordvector/word2vec/GoogleNews-vectors-negative300.bin'
        Word2Vec_embedding = self.load_word_embedding_from_gensim(input_w2v_path)
        word_list = ['PADDING','<e>','<unk>']
        embedding_dic = {}
        embedding_dic['PADDING'] = np.array([0.0] * self.emb_dim)
        embedding_dic['<e>'] = np.random.uniform(low=-0.25, high=0.25, size=(self.emb_dim,))
        embedding_dic['<unk>'] = np.random.uniform(low=-0.25, high=0.25, size=(self.emb_dim,))
        print('Dump word embedding to', self.word_embedding_path)
        with open(self.word_embedding_path, 'w') as outfile:
            for word in word_list:
                outfile.write(word+' ')
                outfile.write(' '.join(str(v) for v in embedding_dic[word]))
                outfile.write('\n')
            for word in list(word_set):
                if word in Word2Vec_embedding:
                    outfile.write(word+' ')
                    outfile.write(' '.join(str(v) for v in Word2Vec_embedding[word]))
                    outfile.write('\n')
                else:
                    exception_counter += 1
        print(f'{exception_counter} words not found.') #207

        # store relation_dic, relation_embedding
        exception_counter = 0
        rela_list = list(rela_set)
        rela_embedding = np.random.uniform(low=-0.25, high=0.25, size=(len(rela_set), self.emb_dim,))
        print('Dump relation embedding to', self.rela_embedding_path)
        with open(self.rela_embedding_path, 'w') as outfile:
            for idx, relation in enumerate(rela_list):
                try:
                    outfile.write(relation+' ')
                    outfile.write(' '.join(str(v) for v in rela_embedding[idx]))
                    outfile.write('\n')
                except:
                    exception_counter += 1
                    print(f'"{relation}" got exception!')
        print(f'{exception_counter} relations not found.') 
        return 0

    def tokenize_train_data(self, data):
        token_data = []
        #start = time.time()
        for idx, q_data in enumerate(data):
            token_q_data = []
            question = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], q_data[0][0]))
            for data_obj in q_data:
                pos_relas = list(map(lambda x: self.rela_dic[x], data_obj[1]))
                pos_words = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[2]))
                neg_relas = list(map(lambda x: self.rela_dic[x], data_obj[3]))
                neg_words = list(map(lambda x: self.word_dic[x] if x in self.word_dic else self.word_dic['<unk>'], data_obj[4]))
                token_q_data.append((question, pos_relas, pos_words, neg_relas, neg_words))
            token_data.append(token_q_data)
        #print(f'Time elapsed:{time.time()-start:.2f}')
        return token_data

    def pad_train_data(self, data, maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w):
    #def pad_train_data(self, data):
        ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        '''
        padded_data = []
        for q_data in data:
            padded_q_data = []
            for obj in q_data:
                q = self.pad_obj(maxlen_q, obj[0])
                p_rela = self.pad_obj(maxlen_pos_r, obj[1])
                p_word = self.pad_obj(maxlen_pos_w, obj[2])
                n_rela = self.pad_obj(maxlen_neg_r, obj[3])
                n_word = self.pad_obj(maxlen_neg_w, obj[4])
                padded_q_data.append((
                    q, p_rela, p_word, n_rela, n_word
                ))
            padded_data.append(padded_q_data)
        return padded_data

    def pad_obj(self, max_len, sentence):
        return [0]*(max_len-len(sentence)) + sentence[:max_len]
        #if max_len >= len(sentence):
        #    return [0]*(max_len-len(sentence)) + sentence
        #else:
        #    return sentence[:max_len]

    def find_maxlength(self, data):
        maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w = 0, 0, 0, 0, 0
        for q_data in data:
            for obj in q_data:
                if len(obj[0]) > maxlen_q:
                    maxlen_q = len(obj[0])
                if len(obj[1]) > maxlen_pos_r:
                    maxlen_pos_r = len(obj[1])
                if len(obj[2]) > maxlen_pos_w:
                    maxlen_pos_w = len(obj[2])
                if len(obj[3]) > maxlen_neg_r:
                    maxlen_neg_r = len(obj[3])
                if len(obj[4]) > maxlen_neg_w:
                    maxlen_neg_w = len(obj[4])
        return maxlen_q, maxlen_pos_r, maxlen_pos_w, maxlen_neg_r, maxlen_neg_w
    
    def load_relations_map(self, path):
        ''' Return self.relations_map = {idx:relation_names}
        '''
        relations_map = {}
        with open(path) as infile:
            idx = 1
            for line in infile:
                relations_map[idx] = line.strip()
                idx += 1
        return relations_map
    
    def question_preprocess(self, path):
        '''Return normalized_question_list from WebQSP
        '''
        q_list = []
        print('Load', path)
        start = time.time()
        WebQSP = json.load(open(path))
        questions = WebQSP['Questions']
        q_list.extend(data['ProcessedQuestion'] for data in questions)
        print(f'Length:{len(q_list)}')
        print(f'Time elapsed:{time.time()-start:.2f}')
        return q_list

def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data


if __name__ == '__main__': 
    data = DataManager()
