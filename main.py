from gensim.utils import simple_preprocess
import numpy as np
from gensim.models import Word2Vec
import os
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
import pickle

import io
import os
import sys

if len(sys.argv) > 2:
    if sys.argv[2] == 'tf':
        os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM, GRU, Reshape, TimeDistributed, Flatten, Dropout, BatchNormalization
from keras.layers import Embedding
from keras.engine import Input
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras.regularizers import l2, activity_l2, l1, activity_l1l2, l1l2
import keras.models

from utils import *
from datetime import datetime

import lxml
import lxml.html
from lxml import etree

if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle

const_is_debug = False
const_is_load_model = False

const_use_all_corpus = True
const_is_prepare_corpus = True
const_corpus = './dataset/Corpus Of Latvian Literature.xml'

const_word2vec_dimensions = 100
const_sentence_length = 10

const_word2vec_epochs = 1000

const_epoch_inner = 10
const_epochs = 100
const_generated_samples = 10
const_batch_size = 512
const_nn_hidden_units = 500

const_l1_regularization = 1e-2
const_learning_rate = 1e-7

if const_is_debug:
    const_word2vec_epochs = 10
    const_epoch_inner = 1

word2idx = {}
idx2word = {}
poems_tokenized = []
word2vec_model = {}

const_nn_type = 'lstm'

if len(sys.argv) > 1:
    const_nn_type = sys.argv[1]

token_empty = u'EMPTY'
token_new_line = u'NEW_LINE'

version = 'v11-' + const_nn_type
init_log('./logs/rnn-{0}-{1}.log'.format(version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
file_csv_loss = open('./results/loss-{}-{}.csv'.format(version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'a')
file_sentences = io.open('./results/file_sentences_{}.txt'.format(version), 'a', encoding='utf-8')

logging.info('version: {}'.format(version))

def prepare_word2vec_model():
    global word2idx, idx2word, poems_tokenized, word2vec_model

    poems = [[]]
    all_corpus = [[]]

    with io.open(const_corpus, 'r', encoding="utf8") as xml_file:
        tree = etree.parse(xml_file)
        el_resources = tree.xpath("//Resource")

        i_poems = 0
        i_resource = 0
        for el_resource in el_resources:
            gendre = el_resource.attrib['Gendre'].lower()

            i_resource += 1

            is_poem = False
            i_poems_per_resource = 0
            if u'dzeja' in gendre:
                is_poem = True
                logging.info(u'i_resource: {} {} {} {}'.format(i_resource, el_resource.attrib['title'],
                                                               el_resource.attrib['author'],
                                                               el_resource.attrib['Gendre']))

            el_paragraphs = el_resource.xpath(".//p")
            for el_paragraph in  el_paragraphs:
                text = lxml.html.fromstring(lxml.html.tostring(el_paragraph)).text_content()
                if text != None and len(text) > 2:
                    text = text[2:-1]

                    is_line_added = False
                    lines = text.split("\n")

                    if is_poem:
                        poems.append([token_empty])

                    if const_use_all_corpus or is_poem:
                        all_corpus.append([token_empty])

                    for line in lines:
                        strip_line = line.strip()
                        if len(strip_line) > 0:
                            processed_line = simple_preprocess(strip_line)
                            processed_line.append(token_new_line)

                            if const_use_all_corpus or is_poem:
                                all_corpus[-1] += processed_line

                            if is_poem and len(lines) > 0:
                                poems[-1] += processed_line
                            is_line_added = True

                    if is_line_added and is_poem:
                        i_poems += 1
                        i_poems_per_resource += 1

            if is_poem:
                logging.info('poems per resource: {}'.format(i_poems_per_resource))



    logging.info('poem count: {}'.format(i_poems))

    logging.info('poem lines count: {}'.format(len(poems)))
    logging.info('corpus lines count: {}'.format(len(all_corpus)))

    word2vec_model = Word2Vec(all_corpus, size=const_word2vec_dimensions, min_count=1, iter=const_word2vec_epochs) #size=100, window=5, min_count=5, workers=4
    word2vec_model.save('word2vec_model.bin')
    word2idx = dict([(k, v.index) for k, v in word2vec_model.vocab.items()])
    idx2word = dict([(v, k) for k, v in word2idx.items()])

    logging.info('Word2Vec built')

    logging.info('word count: {}'.format(len(word2vec_model.vocab.items())))

    cPickle.dump(word2idx, open('word2vec_vocab.bin', 'wb'))

    for poem in poems:
        poem_tokenized = []
        for word in poem:
            poem_tokenized.append(word2idx[word])
        poems_tokenized.append(poem_tokenized)

    cPickle.dump(poems_tokenized, open('poems_tokenized.bin', 'wb'))


def load_word2vec_model():
    global word2idx, idx2word, poems_tokenized, word2vec_model

    word2vec_model = Word2Vec.load('word2vec_model.bin')
    poems_tokenized = cPickle.load(open('poems_tokenized.bin', 'rb'))
    word2idx = cPickle.load(open('word2vec_vocab.bin', 'rb'))
    idx2word = dict([(v, k) for k, v in word2idx.items()])


if const_is_prepare_corpus:
    prepare_word2vec_model()
else:
    load_word2vec_model()

logging.info('model build started')

model = Sequential()

if const_is_load_model and os.path.exists('model-{}.h5'.format(version)):
    model = keras.models.load_model('model-{}.h5'.format(version))
    logging.info('model loaded')
else:
    if const_nn_type == 'lstm':
        model.add(LSTM(output_dim=const_nn_hidden_units, input_dim=const_word2vec_dimensions, input_length=const_sentence_length,
                            return_sequences=True, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
        model.add(LSTM(output_dim=const_nn_hidden_units, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
    elif const_nn_type == 'gru':
        model.add(GRU(output_dim=const_nn_hidden_units, input_dim=const_word2vec_dimensions, input_length=const_sentence_length,
                            return_sequences=True, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
        model.add(GRU(output_dim=const_nn_hidden_units, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
    elif const_nn_type == 'relu':
        model.add(SimpleRNN(output_dim=const_nn_hidden_units, input_dim=const_word2vec_dimensions, input_length=const_sentence_length, activation='relu',
                            return_sequences=True, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
        model.add(SimpleRNN(output_dim=const_nn_hidden_units, W_regularizer=l1l2(const_l1_regularization), activation='relu',
                            U_regularizer=l1l2(const_l1_regularization)))
    else:
        model.add(SimpleRNN(output_dim=const_nn_hidden_units, input_dim=const_word2vec_dimensions, input_length=const_sentence_length,
                            return_sequences=True, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))
        model.add(SimpleRNN(output_dim=const_nn_hidden_units, W_regularizer=l1l2(const_l1_regularization),
                            U_regularizer=l1l2(const_l1_regularization)))

    # model.add(Flatten())
    model.add(Dropout(p=0.3))
    model.add(Dense(output_dim=const_word2vec_dimensions, W_regularizer=l1l2(const_l1_regularization),
                    activity_regularizer=activity_l1l2(const_l1_regularization)))

optimizer = Adam(lr=const_learning_rate)
model.compile(loss='mse', optimizer=optimizer)

logging.info('model build finished')

embeded_token_empty = word2vec_model[token_empty].tolist()

X = []
Y = []
for poem in poems_tokenized:
    prev_words = []
    for word_index in poem:
        word_embeded = word2vec_model[idx2word[word_index]].tolist()
        if len(prev_words) > 0:
            x_so_far = prev_words[:]
            if len(x_so_far) > const_sentence_length:
                x_so_far = prev_words[-const_sentence_length:]

            if len(x_so_far) < const_sentence_length:
                count_missing = const_sentence_length-len(x_so_far)
                list_missing = []
                for _ in range(count_missing):
                    list_missing.append(embeded_token_empty)
                x_so_far = list_missing + x_so_far

            X.append( x_so_far )
            Y.append( word_embeded )

        prev_words.append( word_embeded )

Y = np.array(Y, dtype=np.float32)
X = np.array(X, dtype=np.float32)

logging.info('training samples finished')

i_total_epoch = 0
for i_epoch in range(const_epochs):
    history = model.fit(X, Y, batch_size=const_batch_size, nb_epoch=const_epoch_inner)

    for loss in history.history['loss']:
        i_total_epoch += 1
        file_csv_loss.write('{};{}\n'.format(i_total_epoch, loss))
        logging.info('epoch: {} loss:{}\n'.format(i_total_epoch, loss))
    file_csv_loss.flush()

    file_sentences.write(u'\n\n---- {} ----\n'.format(i_epoch + 1));
    file_sentences.flush()

    model.save('model-{}.h5'.format(version))

    for i_sample in range(const_generated_samples):
        x = []
        for _ in range(const_sentence_length):
            x.append(embeded_token_empty)

        x_words = []

        closest_word = idx2word[np.random.randint(0, len(idx2word)-1)]
        x_n = word2vec_model[closest_word]
        x_words.append(closest_word)

        for n in range(const_sentence_length):
            x.append(x_n)
            x = x[1:]

            y = model.predict(np.reshape(x, (1, const_sentence_length, const_word2vec_dimensions)), batch_size=1)
            closest_word = word2vec_model.most_similar(positive=[np.reshape(y, (const_word2vec_dimensions,))], topn=1)[0][0]

            x_words.append(closest_word)

            x_n = word2vec_model[closest_word]

        file_sentences.write(u'{}\n\n'.format(' '.join(x_words)))
        file_sentences.flush()

        logging.info('sample {} {}'.format(i_sample, x_words))


pass


#https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
# http://stackoverflow.com/questions/34967312/how-to-stack-lstm-layers-to-classify-speech-files