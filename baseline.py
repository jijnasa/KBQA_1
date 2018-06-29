import sys
import time
import argparse
import math
import random
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_preprocess import DataManager
from model import HR_BiLSTM
from model import ABWIM

def batchify(data, batch_size):
    ''' Input: training_data_list [[(question, pos_relas, pos_words, neg_relas, neg_words) * neg_size] * q_size]
        Return: [[(question, pos_relas, pos_words, neg_relas, neg_words)*neg_size] * batch_size] * nb_batch]
    '''
    nb_batch = math.ceil(len(data) / batch_size)
    batch_data = [data[idx*batch_size:(idx+1)*batch_size] for idx in range(nb_batch)]
    print('nb_batch', len(batch_data), 'batch_size', len(batch_data[0]))
    return batch_data

def cal_acc(sorted_score_label):
    if sorted_score_label[0][1] == 1:
        return 1
    else:
        return 0

def save_best_model(model):
    import datetime
    now = datetime.datetime.now()
    if args.save_model_path == '':
        args.save_model_path = f'save_model/{now.month}{now.day}_{now.hour}h{now.minute}m.pt'
        with open('log.txt', 'a') as outfile:
            outfile.write(str(args)+'\n')
    print('save model at {}'.format(args.save_model_path))
    with open(args.save_model_path, 'wb') as outfile:
        torch.save(model, outfile)

def train(args):
    # Build model
    print('Build model')
    if args.model == 'ABWIM':
        q_len = corpus.maxlen_q
        r_len = corpus.maxlen_w + corpus.maxlen_r
        #print('q_len', q_len, 'r_len', r_len)
        model = ABWIM(args.dropout, args.hidden_size, corpus.word_embedding, corpus.rela_embedding, q_len, r_len).cuda()
    elif args.model == 'HR-BiLSTM':
        model = HR_BiLSTM(args.dropout, args.hidden_size, corpus.word_embedding, corpus.rela_embedding).cuda()
    print(model)

    if args.optimizer == 'Adadelta':
        print('optimizer: Adadelta')
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    else:
        print('optimizer: SGD')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    print()
    #print("hello")
    best_model = None
    best_val_loss = None
    train_start_time = time.time()

    earlystop_counter = 0
    global_step = 0

    for epoch_count in range(0, args.epoch_num):
        model.train()

        total_loss, total_acc = 0.0, 0.0
        nb_question = 0
        epoch_start_time = time.time()

        for batch_count, batch_data in enumerate(train_data, 1):
            variable_start_time = time.time()
            if args.batch_type == 'batch_question':
                training_objs = [obj for q_obj in batch_data for obj in q_obj]
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
                nb_question += len(batch_data)
            elif args.batch_type == 'batch_obj':
                question, pos_relas, pos_words, neg_relas, neg_words = zip(*batch_data)
            #print('len(question)', len(question))
            q = Variable(torch.LongTensor(question)).cuda()
            p_relas = Variable(torch.LongTensor(pos_relas)).cuda()
            p_words = Variable(torch.LongTensor(pos_words)).cuda()
            n_relas = Variable(torch.LongTensor(neg_relas)).cuda()
            n_words = Variable(torch.LongTensor(neg_words)).cuda()
            ones = Variable(torch.ones(len(question))).cuda()
            variable_end_time = time.time()
            
            optimizer.zero_grad()
            all_pos_score = model(q, p_relas, p_words)
            all_neg_score = model(q, n_relas, n_words)
            model_end_time = time.time()

            loss = loss_function(all_pos_score, all_neg_score, ones)
            loss.backward()
            optimizer.step()
            loss_backward_time = time.time()
            writer.add_scalar('data/pre_gen_loss', loss.item(), global_step)
            global_step += 1
            if torch.__version__ == '0.3.0.post4':
                total_loss += loss.data.cpu().numpy()[0]
            else:
                total_loss += loss.data.cpu().numpy()
            average_loss = total_loss / batch_count

            # Calculate accuracy and f1
            if args.batch_type == 'batch_question':
                all_pos = all_pos_score.data.cpu().numpy()
                all_neg = all_neg_score.data.cpu().numpy()
                start, end = 0, 0
                for idx, q_obj in enumerate(batch_data):
                    end += len(q_obj)
                    score_list = [all_pos[start]]
                    #print('len(score_list), score_list')
                    #print(len(score_list), score_list)
                    batch_neg_score = all_neg[start:end]
                    start = end
                    label_list = [1]
                    for ns in batch_neg_score:
                        score_list.append(ns)
                    label_list += [0] * len(batch_neg_score)
                    #print('len(score_list), score_list')
                    #print(len(score_list), score_list)
                    #print('len(label_list), label_list')
                    #print(len(label_list), label_list)
                    score_label = [(x, y) for x, y in zip(score_list, label_list)]
                    sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
                    total_acc += cal_acc(sorted_score_label)
                #average_acc = total_acc / (batch_count * args.batch_size)
                average_acc = total_acc / nb_question
                elapsed = time.time() - epoch_start_time
                print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} #_question:{nb_question}'
            else:
                elapsed = time.time() - epoch_start_time
                print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f}'

            #print(f'variable time      :{variable_end_time-variable_start_time:.3f} / {variable_end_time-epoch_start_time:.3f}')
            #print(f'model time         :{model_end_time - variable_end_time:.3f} / {model_end_time-epoch_start_time:.3f}')
            #print(f'loss calculate time:{loss_backward_time-model_end_time:.3f} / {loss_backward_time-epoch_start_time:.3f}')

            #print_str = f'Epoch {epoch_count} batch {batch_count} Spend Time:{elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} #_question:{nb_question}'
            if batch_count % 10 == 0:
                print('\r', print_str, end='')
            #batch_end_time = time.time()
            #print('one batch', batch_end_time-batch_start_time)
        print('\r', print_str, end='')
        print()
        val_print_str, val_loss, _ = evaluation(model, 'dev', global_step)
        print('Val', val_print_str)
        #log_str, _, test_acc = evaluation(model, 'test')
        #print('Test', log_str)
        #print('Test Acc', test_acc)

        # this section handle earlystopping
        if not best_val_loss or val_loss < best_val_loss:
            earlystop_counter = 0
            best_model = model
            save_best_model(best_model)
            best_val_loss = val_loss
        else:
            earlystop_counter += 1
        if earlystop_counter >= args.earlystop_tolerance:
            print('EarlyStopping!')
            print(f'Total training time {time.time()-train_start_time:.2f}')
            break
    return best_model

def evaluation(model, mode='dev', global_step=None):
    model_test = model.eval()
    start_time = time.time()
    total_loss, total_acc = 0.0, 0.0
    if mode == 'test':
        input_data = test_data
        #print(model_test)
    else:
        input_data = val_data
    nb_question = sum(len(batch_data) for batch_data in input_data)
    #print('nb_question', nb_question)
    
    for batch_count, batch_data in enumerate(input_data, 1):
        training_objs = [obj for q_obj in batch_data for obj in q_obj]
        question, pos_relas, pos_words, neg_relas, neg_words = zip(*training_objs)
        #print(question[:5])
        #print(pos_relas[:5])
        #print(pos_words[:5])
        #print(neg_relas[:5])
        #print(neg_words[:5])
        q = Variable(torch.LongTensor(question)).cuda()
        p_relas = Variable(torch.LongTensor(pos_relas)).cuda()
        p_words = Variable(torch.LongTensor(pos_words)).cuda()
        n_relas = Variable(torch.LongTensor(neg_relas)).cuda()
        n_words = Variable(torch.LongTensor(neg_words)).cuda()
        ones = Variable(torch.ones(len(question))).cuda()
        
        pos_score = model_test(q, p_relas, p_words)
        neg_score = model_test(q, n_relas, n_words)
        loss = loss_function(pos_score, neg_score, ones)
        if torch.__version__ == '0.3.0.post4':
            total_loss += loss.data.cpu().numpy()[0]
        else:
            total_loss += loss.data.cpu().numpy()
        average_loss = total_loss / batch_count

        # Calculate accuracy and f1
        all_pos = pos_score.data.cpu().numpy()
        all_neg = neg_score.data.cpu().numpy()
        start, end = 0, 0
        for idx, q_obj in enumerate(batch_data):
            end += len(q_obj)
            #print('start', start, 'end', end)
            score_list = [all_pos[start]]
            label_list = [1]
            batch_neg_score = all_neg[start:end]
            for ns in batch_neg_score:
                score_list.append(ns)
            label_list += [0] * len(batch_neg_score)
            start = end
            score_label = [(x, y) for x, y in zip(score_list, label_list)]
            #print(score_label[:10])
            #print('len(score_list)', len(score_list), 'len(label_list)', len(label_list), 'len(score_label)', len(score_label))
            sorted_score_label = sorted(score_label, key=lambda x:x[0], reverse=True)
            #print(sorted_score_label)
            total_acc += cal_acc(sorted_score_label)
            #print(total_acc)
            #input('Enter')

#        acc1 = total_acc / (batch_count * args.batch_size)
#        acc2 = total_acc / question_counter

    if mode == 'dev':
        writer.add_scalar('val_loss', average_loss.item(), global_step)

    time_elapsed = time.time()-start_time
    average_acc = total_acc / nb_question
#    print('acc1', acc1)
#    print('acc2', acc2)
#    print('average_acc', average_acc)
#    print(question_counter, nb_question)
    print_str = f'Batch {batch_count} Spend Time:{time_elapsed:.2f}s Loss:{average_loss*1000:.4f} Acc:{average_acc:.4f} # question:{nb_question}'
    return print_str, average_loss, average_acc

if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    # setting
    parser.add_argument('-train', default=False, action='store_true')
    parser.add_argument('-test', default=False, action='store_true')
    parser.add_argument('--model', type=str, required=True) # [ABWIM/HR-BiLSTM]
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=2.0) # [0.1/0.5/1.0/2.0]
    parser.add_argument('--hidden_size', type=int, default=100) # [50/100/200/400]
    parser.add_argument('--optimizer', type=str, default='Adadelta')
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--batch_type', type=str, default='batch_question') # [batch_question/batch_obj]
    parser.add_argument('--batch_question_size', type=int, default=32)
    parser.add_argument('--batch_obj_size', type=int, default=128)
    parser.add_argument('--earlystop_tolerance', type=int, default=5)
    parser.add_argument('--save_model_path', type=str, default='')
    parser.add_argument('--pretrain_model', type=str, default=None)
    args = parser.parse_args()
    if args.model == 'ABWIM':
        args.margin = 0.1
        args.optimizer = 'Adadelta'
    loss_function = nn.MarginRankingLoss(margin=args.margin)

    # Load data
    corpus = DataManager()
    if args.train:
        # shuffle training data
        random.shuffle(corpus.token_train_data)
        # split training data to train and validation
        split_num = int(0.9*len(corpus.token_train_data))
        print('split_num=', split_num)
        train_data = corpus.token_train_data[:split_num]
        val_data = corpus.token_train_data[split_num:]
        print('training data length:', len(train_data))
        print('validation data length:', len(val_data))

        if args.batch_type == 'batch_question':
            # batchify questions, uncomment Line 119, 120
            train_data = batchify(train_data, args.batch_question_size)
        elif args.batch_type == 'batch_obj':
            # batchify train_objs, uncomment Line 121
            flat_train_data = [obj for q_obj in train_data for obj in q_obj]
            print('len(flat_train_data)', len(flat_train_data))
            random.shuffle(flat_train_data)
            train_data = batchify(flat_train_data, args.batch_obj_size)
        val_data = batchify(val_data, args.batch_question_size)

        # Create SummaryWriter
        writer = SummaryWriter(log_dir='save_model/tensorboard_log')
        train(args)

    if args.test:
        print('test data length:', len(corpus.token_test_data))
        test_data = batchify(corpus.token_test_data, args.batch_question_size)
        if args.pretrain_model == None:
            print('Load best model', args.save_model_path)
            with open(args.save_model_path, 'rb') as infile:
                model = torch.load(infile)
        else:
            print('Load pretrain model', args.pretrain_model)
            with open(args.pretrain_model, 'rb') as infile:
                model = torch.load(infile) 
        log_str, _, test_acc = evaluation(model, 'test')
        print(log_str)
        print(test_acc)
        with open('log.txt', 'a') as outfile:
            if args.pretrain_model == None:
                outfile.write(str(test_acc)+'\t'+args.save_model_path+'\n')
            else:
                outfile.write(str(test_acc)+'\t'+args.pretrain_model+'\n')

    # Close writer
    writer.close()

