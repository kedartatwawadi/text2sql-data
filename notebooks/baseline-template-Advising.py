
# coding: utf-8

# In[17]:


import argparse
from collections import Counter
import pickle
import random
import json
import sys
import numpy as np
import dynet as dy # Loaded late to avoid memory allocation when we just want help info


# In[18]:


## Command line argument handling and default configuration ##

abstract_lang = 'sql'
#--eval_freq 1000000 --log_freq 1000000 --max_bad_iters -1 --do_test_eval

import sqlite3
conn = sqlite3.connect('../data/advising-sqlite.db')
c = conn.cursor()


# In[19]:


parser = argparse.ArgumentParser(description='A simple template-based text-to-SQL system.')

parser.add_argument('data', help='Data in json format', nargs='+')
parser.add_argument('--unk_max', help='Maximum count to be considered an unknown word', type=int, default=0)
parser.add_argument('--query_split', help='Use the query split rather than the question split', action='store_true')
parser.add_argument('--no_vars', help='Run without filling in variables', action='store_true')
parser.add_argument('--use_all_sql', help='Default is to use first SQL only, this makes multiple instances.', action='store_true')
parser.add_argument('--split', help='Use this split in cross-validation.', type=int) # Used for small datasets: Academic, Restaurants, IMDB, Yelp

# Model
parser.add_argument('--mlp', help='Use a multi-layer perceptron', action='store_true')
parser.add_argument('--dim_word', help='Dimensionality of word embeddings', type=int, default=128)
parser.add_argument('--dim_hidden_lstm', help='Dimensionality of LSTM hidden vectors', type=int, default=64)
parser.add_argument('--dim_hidden_mlp', help='Dimensionality of MLP hidden vectors', type=int, default=32)
parser.add_argument('--dim_hidden_template', help='Dimensionality of MLP hidden vectors for the final template choice', type=int, default=64)
parser.add_argument('--word_vectors', help='Pre-built word embeddings')
parser.add_argument('--lstm_layers', help='Number of layers in the LSTM', type=int, default=3)

# Training
parser.add_argument('--max_iters', help='Maximum number of training iterations', type=int, default=40)
parser.add_argument('--max_bad_iters', help='Maximum number of consecutive training iterations without improvement', type=int, default=1)
parser.add_argument('--log_freq', help='Number of examples to decode between logging', type=int, default=100000)
parser.add_argument('--eval_freq', help='Number of examples to decode between evaluation runs', type=int, default=100000)
parser.add_argument('--train_noise', help='Noise added to word embeddings as regularization', type=float, default=0.1)
parser.add_argument('--lstm_dropout', help='Dropout for input and hidden elements of the LSTM', type=float, default=0.0)
parser.add_argument('--learning_rate', help='Learning rate for optimiser', type=float, default=0.1)

args = parser.parse_args()


# In[20]:


## Input ##

def insert_variables(sql, sql_variables, sent, sent_variables):
    tokens = []
    tags = []
    for token in sent.strip().split():
        if (token not in sent_variables) or args.no_vars:
            tokens.append(token)
            tags.append("O")
        else:
            assert len(sent_variables[token]) > 0
            for word in sent_variables[token].split():
                tokens.append(word)
                tags.append(token)

    sql_tokens = []
    for token in sql.strip().split():
        if token.startswith('"%') or token.startswith("'%"):
            sql_tokens.append(token[:2])
            token = token[2:]
        elif token.startswith('"') or token.startswith("'"):
            sql_tokens.append(token[0])
            token = token[1:]

        if token.endswith('%"') or token.endswith("%'"):
            sql_tokens.append(token[:-2])
            sql_tokens.append(token[-2:])
        elif token.endswith('"') or token.endswith("'"):
            sql_tokens.append(token[:-1])
            sql_tokens.append(token[-1])
        else:
            sql_tokens.append(token)

    template = []
    for token in sql_tokens:
        if (token not in sent_variables) and (token not in sql_variables):
            template.append(token)
        elif token in sent_variables:
            if sent_variables[token] == '':
                example = None
                for variable in sql_variables:
                    if variable['name'] == token:
                        example = variable['example']
                assert example is not None
                template.append(example)
            else:
                template.append(token)
        elif token in sql_variables:
            example = None
            for variable in sql_variables:
                if variable['name'] == token:
                    example = variable['example']
            assert example is not None
            template.append(example)
            
    template_tags = sorted(list(set(tags)))
    return (tokens, tags, ' '.join(template), sent_variables )

def get_tagged_data_for_query(data):
    dataset = data['query-split']
    for sent_info in data['sentences']:
        if not args.query_split:
            dataset = sent_info['question-split']

        if args.split is not None:
            if str(args.split) == str(dataset):
                dataset = "test"
            else:
                dataset = "train"

        for sql in data[abstract_lang]:
            sql_vars = data['variables']
            text = sent_info['text']
            text_vars = sent_info['variables']

            yield (dataset, insert_variables(sql, sql_vars, text, text_vars))

            if not args.use_all_sql:
                break


# In[21]:



# In[22]:


## Set up voacbulary ##

class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        for word in corpus:
            w2i.setdefault(word, len(w2i))
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())

def build_vocab(sentences):
    counts = Counter()
    words = {"<UNK>"}
    tag_set = set()
    template_set = set()
    template_to_tagset = {}
    for tokens, tags, template, text_variables in train:
        template_set.add(template)
        template_to_tagset[template] = set(text_variables.keys())
        template_to_tagset[template].add('O')
        for tag in tags:
            tag_set.add(tag)
        for token in tokens:
            counts[token] += 1

    for word in counts:
        if counts[word] > args.unk_max:
            words.add(word)

    vocab_tags = Vocab.from_corpus(tag_set)
    vocab_words = Vocab.from_corpus(words)
    vocab_templates = Vocab.from_corpus(template_set)

    return vocab_words, vocab_tags, vocab_templates, template_to_tagset


# In[23]:

train = []
dev = []
test = []
for filename in args.data:
    with open(filename) as input_file:
        data = json.load(input_file)
        if type(data) == list:
            for example in data:
                for dataset, instance in get_tagged_data_for_query(example):
                    if dataset == 'train':
                        train.append(instance)
                    elif dataset == 'dev':
                        train.append(instance)
                    elif dataset == 'test':
                        test.append(instance)
                    elif dataset == 'exclude':
                        pass
                    else:
                        assert False, dataset
        else:
            for dataset, instance in get_tagged_data_for_query(data):
                if dataset == 'train':
                    train.append(instance)
                elif dataset == 'dev':
                    train.append(instance)
                elif dataset == 'test':
                    test.append(instance)
                elif dataset == 'exclude':
                    pass
                else:
                    assert False, dataset


vocab_words, vocab_tags, vocab_templates,template_to_tagset = build_vocab(train)
UNK = vocab_words.w2i["<UNK>"]
NWORDS = vocab_words.size()
NTAGS = vocab_tags.size()
NTEMPLATES = vocab_templates.size()

print("Running with {} templates".format(NTEMPLATES))


# In[24]:


template_set_test = set()
count = 0
for tokens, tags, template, text_variables in test:
    if template not in template_to_tagset:
        template_set_test.add(template)
        template_to_tagset[template] = set(text_variables.keys())
        template_to_tagset[template].add('O')
        count += 1
print(count)


# In[25]:


## Set up model ##

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=args.learning_rate)
DIM_WORD = args.dim_word
DIM_HIDDEN_LSTM = args.dim_hidden_lstm
DIM_HIDDEN_MLP = args.dim_hidden_mlp
DIM_HIDDEN_TEMPLATE = args.dim_hidden_template

pEmbedding = model.add_lookup_parameters((NWORDS, DIM_WORD))
if args.word_vectors is not None:
    pretrained = []
    with open(args.word_vectors,'rb') as pickleFile:
        embedding = pickle.load(pickleFile)
        for word_id in range(vocab_words.size()):
            word = vocab_words.i2w[word_id]
            if word in embedding:
                pretrained.append(embedding[word])
            else:
                pretrained.append(pEmbedding.row_as_array(word_id))
    pEmbedding.init_from_array(np.array(pretrained))
if args.mlp:
    pHidden = model.add_parameters((DIM_HIDDEN_MLP, DIM_HIDDEN_LSTM*2))
    pOutput = model.add_parameters((NTAGS, DIM_HIDDEN_MLP))
else:
    pOutput = model.add_parameters((NTAGS, DIM_HIDDEN_LSTM*2))

builders = [
    dy.LSTMBuilder(args.lstm_layers, DIM_WORD, DIM_HIDDEN_LSTM, model),
    dy.LSTMBuilder(args.lstm_layers, DIM_WORD, DIM_HIDDEN_LSTM, model),
]

pHiddenTemplate = model.add_parameters((DIM_HIDDEN_TEMPLATE, DIM_HIDDEN_LSTM*2))
pOutputTemplate = model.add_parameters((NTEMPLATES, DIM_HIDDEN_TEMPLATE))


# In[26]:


def build_tagging_graph(words, tags, template, builders, train=True,k=1):
    dy.renew_cg()

    if train and args.lstm_dropout is not None and args.lstm_dropout > 0:
        for b in builders:
            b.set_dropouts(args.lstm_dropout, args.lstm_dropout)

    f_init, b_init = [b.initial_state() for b in builders]

    wembs = [dy.lookup(pEmbedding, w) for w in words]
    if train: # Add noise in training as a regularizer
        wembs = [dy.noise(we, args.train_noise) for we in wembs]

    fw_states = [x for x in f_init.add_inputs(wembs)]
    bw_states = [x for x in b_init.add_inputs(reversed(wembs))]
    fw = [x.output() for x in fw_states]
    bw = [x.output() for x in bw_states]

    O = dy.parameter(pOutput)
    if args.mlp:
        H = dy.parameter(pHidden)
    errs = []
    pred_tags = []
    sorted_arg_topk = []
    final_topk = []
    sequences_topk = [(0.0,list())]
    for f, b, t in zip(fw, reversed(bw), tags):
        f_b = dy.concatenate([f,b])
        if args.mlp:
            f_b = dy.tanh(H * f_b)
        r_t = O * f_b
        if train:
            err = dy.pickneglogsoftmax(r_t, t)
            errs.append(err)
        else:
            out = dy.log_softmax(r_t)
            chosen = np.argmax(out.npvalue())
            pred_tags.append(vocab_tags.i2w[chosen])
            all_sequences = list()
            for seq in sequences_topk:
                seq_score,seq_list = seq
                _scores = -out.npvalue()
                arg_topk = np.argsort(_scores)[:k]
                score_topk = _scores[arg_topk]
                for i in range(min(k,len(arg_topk))):
                    _list = list(seq_list)
                    _list.append(vocab_tags.i2w[arg_topk[i]])
                    score = seq_score + score_topk[i]
                    all_sequences.append((score,_list))
            sequences_topk = sorted(all_sequences)[:k]
            

    O_template = dy.parameter(pOutputTemplate)
    H_template = dy.parameter(pHiddenTemplate)
    f_bt = dy.concatenate([fw_states[-1].s()[0], bw_states[-1].s()[0]])
    f_bt = dy.tanh(H_template * f_bt)
    r_tt = O_template * f_bt
    pred_template = None
    if train:
        err = dy.pickneglogsoftmax(r_tt, template)
        errs.append(err)
    else:
        out = dy.log_softmax(r_tt)
        _scores = -out.npvalue()
        chosen = np.argmin(_scores)        
        pred_template = vocab_templates.i2w[chosen]
        sorted_arg_topk = np.argsort(_scores)[:k]
        
        all_sequences_and_templates = []
        for template_id in sorted_arg_topk:
            _score = _scores[template_id]
            _template = vocab_templates.i2w[template_id]

            for seq_score,seq_list in sequences_topk:
                all_sequences_and_templates.append((_score + seq_score, seq_list, _template))
        final_topk = sorted(all_sequences_and_templates)[:k]    
        
        
    return pred_tags, pred_template, errs, final_topk


# In[27]:


tagged = 0
loss = 0
best_dev_acc = 0.0
iters_since_best_updated = 0
steps = 0
for iteration in range(args.max_iters):
    random.shuffle(train)
    for tokens, tags, template, text_variables in train:
        steps += 1

        # Convert to indices
        word_ids = [vocab_words.w2i.get(word, UNK) for word in tokens]
        tag_ids = [vocab_tags.w2i[tag] for tag in tags]
        template_id = vocab_templates.w2i[template]

        # Decode and update
        _, _, errs,_ = build_tagging_graph(word_ids, tag_ids, template_id, builders)
        sum_errs = dy.esum(errs)
        loss += sum_errs.scalar_value()
        tagged += len(tag_ids)
        sum_errs.backward()
        trainer.update()

        # Log status
        if steps % args.log_freq == 0:
            trainer.status()
            print("TrainLoss {}-{}: {}".format(iteration, steps, loss / tagged))
            loss = 0
            tagged = 0
            sys.stdout.flush()
        if steps % args.eval_freq == 0:
            acc = run_eval(dev, builders, iteration, steps)
            if best_dev_acc < acc:
                best_dev_acc = acc
                iters_since_best_updated = 0
                print("New best Acc!", acc)
            sys.stdout.flush()
    iters_since_best_updated += 1
    if args.max_bad_iters > 0 and iters_since_best_updated > args.max_bad_iters:
        print("Stopping at iter {} as there have been {} iters without improvement".format(iteration, args.max_bad_iters))
        break



# In[28]:


def get_sql_from_token_template(_tokens,_tags,template):
    sql = str(template)
    tokens = list(_tokens) 
    tags = list(_tags)
    tokens.append("test")
    tags.append("O")
    _curr_tag = ""
    _curr_token = ""
    for _token,_tag in zip(tokens,tags):
        if _tag is _curr_tag:
            _curr_token = _curr_token + " " + _token
        else:
            if _curr_tag is 'O':
                pass
            else:
                if _curr_token.isdigit():
                    sql = sql.replace(_curr_tag,_curr_token)
                else:
                    sql = sql.replace(" " + _curr_tag + " ",_curr_token)
            _curr_tag = _tag
            _curr_token = _token
    return sql


# In[29]:


def run_eval(data, builders, iteration, step,k):
    if len(data) == 0:
        print("No data for eval")
        return -1
    good = 0.0
    total = 0.0
    complete_good = 0.0
    templates_good = 0.0
    oracle = 0.0
    tag_set_wrong = 0.0
    inconsistent_examples = 0
    template_error = 0.0
    tags_good = 0.0
    for tokens, tags, template, text_variables in data:
        word_ids = [vocab_words.w2i.get(word, UNK) for word in tokens]
        tag_ids = [0 for tag in tags]
        pred_tags, pred_template, _ , final_topk = build_tagging_graph(word_ids, tag_ids, 0, builders, False,k=k)
        gold_tags = tags
        
        inconsistent = True
        only_test_template = False
        if template in template_set_test:
            only_test_template = True
            template_error += 1.0
        else:
            _temp = template_to_tagset[template]
            if _temp != set(gold_tags):
                inconsistent_examples += 1.0
            else:
                inconsistent = False
        
        for score, _tags, _template in final_topk:
            if template_to_tagset[_template] == set(_tags):
                exception = 0
                try:
                    sql_query = get_sql_from_token_template(tokens,_tags,_template)
                    c.execute(sql_query)
                    out = c.fetchall()
#                     if(len(out) == 0):
#                         exception = 1
#                         print(out)
#                         print(text_variables)
#                         print(tokens)
#                         print(_tags)
#                         print(_template)
#                         print()
                except Exception as e:
                    exception = 1
                    out = []
                
                if not exception:
                    pred_tags = _tags
                    pred_template = _template
                    break;
        
        tags_correct = True
        for gold, pred in zip(gold_tags, pred_tags):
            total += 1
            if gold == pred: 
                good += 1
            else:
                tags_correct = False
                
        if gold_tags == pred_tags:
            tags_good += 1
        
        tag_set_correct = True
        if template_to_tagset[pred_template] != set(pred_tags):
#             print(template_to_tagset[pred_template])
#             print(pred_tags)
            tag_set_wrong += 1
            tag_set_correct = False
        
        template_correct = False
        if pred_template == template:
            templates_good += 1
            template_correct = True
        
        
        if tag_set_correct and template_correct:
            complete_good += 1
        
        if template in vocab_templates.w2i:
            oracle += 1
    tok_acc = good / total
    
    print(len(data))
    tagset_acc = 1.0 - tag_set_wrong/ len(data)
    complete_acc = complete_good / len(data)
    template_acc = templates_good / (len(data) - template_error)
    tag_acc = tags_good/len(data)
    print("Inconsistent for copy mechanism: ", inconsistent_examples*1.0/len(data))
    print("Template only in Test data: ", template_error*1.0/len(data))
    print("Tok-Acc: {:>5} Tags: {:<5} Tmpl: {:<5} Tot: {:>5} Tag: {:>5}".format( tok_acc, tag_acc, template_acc, complete_acc, tagset_acc))
    return complete_acc


# In[31]:


run_eval(test[:100], builders, "End", "test",1)


# In[32]:


for k in range(1,10,2):
    run_eval(test, builders, "End", "test",k)


# In[16]:


print(len(test))

