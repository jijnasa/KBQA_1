import sys
import json
import subprocess
import pickle
import random

exe = '/home/zychen/KBQA-GAN/FastRDFStore/bin/FastRDFStoreClient.exe'
server = '140.109.19.67'

def find_repeat_mid(path):
    topic_mids = set()
    ans_mids = set()
    data = json.load(open(path))
    for qa in data['Questions']:
        for info in qa['Parses']:
            topic_mids.add(info['TopicEntityMid'])
            for ans in info['Answers']:
                ans_mids.add(ans['AnswerArgument'])

    repeat = []
    topic_mid_list = list(topic_mids)
    for mid in topic_mid_list:
        if mid in ans_mids:
            repeat.append(mid)
    print(len(repeat)) # 427
    return repeat

def process_result(result):
#    print(result)
    result = result.rstrip()
    result_list = [x for x in result.rstrip().split('\n')]
    result_list = result_list[1:-1]
    
    rela_list = []
    topic = 'unk'

    for idx, line in enumerate(result_list):
#        print(idx, line)
        if 'common.document.text' in line:
            print('Skip: common.document.text')
            continue
        if line[0] != ' ':
            line = line.replace(' ','').replace('-->','(').replace(')', '')
            tokens = line.split('(')
            if len(tokens) == 3 and ('m.' in tokens[2] or 'g.' in tokens[2]):
                rela = tokens[0]
                name = tokens[1]
                mid = tokens[2]
                rela_list.append((rela, name, mid))
            elif 'type.object.name' in tokens[0]:
                topic = tokens[1]
            else:
                print('Skip:', line)

    print('topic', topic, 'len(rela_list)', len(rela_list))    
#    for i, r in enumerate(rela_list):
#        print(i, r)

    return topic, rela_list

def query(mid):
    print('query:', mid)
    p = subprocess.Popen([exe, '-m', mid, '-s', server],
                    stdout=subprocess.PIPE, encoding="utf-8")
    #print('s', s)
    #print('waiting for subprocess return')
    outs, _ = p.communicate()
#    result = outs
#    result_list = process_result(result)
    topic, rela_list = process_result(outs)
#    rela_list = [x for x in result_list if len(x) > 0]
    return topic, rela_list

def recursive(mid, depth):
    print('recursive depth', depth)
    topic, rela_list = query(mid)
    if len(rela_list) < 1 or depth == 0:
        return []
    i = random.randrange(len(rela_list))
    path = recursive(rela_list[i][2], depth-1)
#    print(type(path), len(path))
    path.append((rela_list[i]))
#    print(type(path), len(path))
    return path

def gen_question_dic(q_path):
    q_dic = {}
    q_dic['RawQuestion']=''
    q_dic['TopicEntityName'] = q_path[0][0]
    q_dic['TopicEntityMid'] = q_path[0][1]
    q_dic['InferentialChain'] = q_path[1:]
    q_dic['AnswerEntityName'] = q_path[-1][1]
    q_dic['AnswerEntityMid'] = q_path[-1][2]
    q_dic['drop'] = False
    entity_set = set()
    mid_set = set()
    entity_set.add(q_path[0][0])
    mid_set.add(q_path[0][1])
    for i in range(1, len(q_path)-1):
        if q_path[i][1] in entity_set:
            q_dic['drop'] = True
        else:
            entity_set.add(q_path[i][1])
        if q_path[i][2] in mid_set:
            q_dic['drop'] = True
        else:
            mid_set.add(q_path[i][2])
    return q_dic

def test_function():
    path_depth = 4 # 4 entities with 3 relations in between
    question_list = []

    mid = 'm.05f7zp9'
    topic,_ = query(mid)
    q_path = recursive(mid, path_depth)
#    print('len(q_path)', len(q_path))
    q_path.append((topic, mid))
    q_path.reverse()
    print(q_path)
    if len(q_path) == path_depth+1:
        question_info = gen_question_dic(q_path)
        print(question_info['drop'])
        if question_info['drop'] == False:
            question_list.append(question_info)
    print(question_list)
#    for idx, entity in enumerate(path):
#        print(idx, entity)

#    name, rela_list, name_list, mid_list = query(mid)
#    print(len(rela_list))
#    print(rela_list)
    

if __name__ == '__main__':
#    test_function()
#    sys.exit()

#    path = '/home/sky51008/dataset/KBQA/WebQuestion/WebQSP/data/WebQSP.train.json'
#
#    repeated_mids = find_repeat_mid(path)
#
#    mids = set()
#    data = json.load(open(path))
#    for qa in data['Questions']:
#        for info in qa['Parses']:
#            mids.add(info['TopicEntityMid'])
#            for ans in info['Answers']:
#                mids.add(ans['AnswerArgument'])
##    print(len(mids)) # 29330
#    with open('/home/ypc/workspace/mids.pkl','wb') as outfile:
#        pickle.dump(mids, outfile)
    with open('/home/ypc/workspace/mids.pkl','rb') as infile:
        mids = pickle.load(infile)

    mid_list = list(mids)
    path_depth = 5 # 5 entities with 4 relations in between, but ignore the input entity
    question_list = []
    for idx, mid in enumerate(mid_list):
        print('idx', idx)
        topic,_ = query(mid)
        q_path = recursive(mid, path_depth)
        q_path.append((topic, mid))
        q_path.reverse()
        if len(q_path) == path_depth+1:
            question_info = gen_question_dic(q_path)
            if question_info['drop'] == False:
                question_list.append(question_info)
        if len(question_list) == 10:
            break
    print(len(question_list))

    with open('/home/ypc/workspace/3rela_questions_7.json', 'w') as outfile:
        json.dump(question_list, outfile, indent=4)

