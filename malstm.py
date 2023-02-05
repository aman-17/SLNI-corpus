import csv
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import math

lemmatizer = WordNetLemmatizer()
max_len = 100


from tensorflow.contrib.rnn import LSTMCell
class SiameseLSTM():
    def __init__(self, config, sess, embeddings is_training=True):
        
        num_step = config['max_len']
        emb_dim = config['emb_dim']
        vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        
        self.sentence_A = tf.placeholder(tf.int32, [None, num_step])
        self.sentence_B = tf.placeholder(tf.int32, [None, num_step])
        self.mask_A = tf.placeholder(tf.float64, [None, num_step])
        self.mask_B = tf.placeholder(tf.float64, [None, num_step])
        self.relatedness = tf.placeholder(tf.float64, [None])
        
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        with tf.name_scope('Embedding_Layer'):
            embedding_initializer = tf.constant_initializer(embeddings, dtype=tf.float64)
            embedding_weights = tf.get_variable(dtype=tf.float64, name='embedding_weights',shape=(vocab_size, emb_dim), initializer=embedding_initializer, trainable=False)
            self.embedded_A = tf.nn.embedding_lookup(embedding_weights, self.sentence_A) # (batch_size, num_step, emb_dim)
            self.embedded_B = tf.nn.embedding_lookup(embedding_weights, self.sentence_B) # (batch_size, num_step, emb_dim)
            
            
        with tf.name_scope('LSTM_Output'):
            self.outputs_A = self.LSTM(sequence=self.embedded_A, reuse=None) # (batch_size, num_step, emb_dim)
            self.outputs_B = self.LSTM(sequence=self.embedded_B, reuse=True) # (batch_size, num_step, emb_dim)
            self.masked_outputs_A = tf.reduce_sum(self.outputs_A * self.mask_A[:, :, None], axis=1) # (batch_size, emb_dim)
            self.masked_outputs_B = tf.reduce_sum(self.outputs_B * self.mask_B[:, :, None], axis=1) # (batch_size, emb_dim)
            
        with tf.name_scope('Similarity'):
            self.diff = tf.reduce_sum(tf.abs(tf.subtract(self.masked_outputs_A, self.masked_outputs_B)), axis=1) # 32
            self.similarity = tf.exp(-1.0 * self.diff)
        
        with tf.name_scope('Loss'):
            diff = tf.subtract(self.similarity, self.relatedness - 1.0) / 4.0 # 32
            self.loss = tf.square(diff) # (batch_size,)
            self.cost = tf.reduce_mean(self.loss) # (1,)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        
        train_variables = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), config['max_grad_norm'])
        
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6)
        
        with tf.name_scope('Train'):
            self.train_op = optimizer.apply_gradients(zip(gradients, train_variables))
        
        
            
    def LSTM(self, sequence, reuse=None):
        def sequence_length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            return tf.cast(length, tf.int32)
            
        with tf.variable_scope('LSTM', reuse=reuse, dtype=tf.float64):
            cell = LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        
        seq_len = sequence_length(sequence)
        with tf.name_scope('Siamese'), tf.variable_scope('Siamese', dtype=tf.float64):
            outputs, state = tf.nn.dynamic_rnn(cell, sequence, dtype=tf.float64, sequence_length=seq_len)
        
        return outputs
    

def datapreprocess(sentence):
    return [lemmatizer.lemmatize(word) for word in word_tokenize(sentence.lower())]  

def adding_mask(sentence):
        pad = np.zeros((max_len))
        pad[len(sentence) - 1] = 1
        for i in range(0, max_len - len(sentence)):
            sentence.append('<unknown>')
        
        return np.array(sentence), pad

def load_dataset():    
    data_file = './sick.csv'
    with open(data_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        sentences_A, sentences_B, relatedness, masks_A, masks_B =  [], [], [], [], []
        for entries in csv_reader:
            sentence_A, mask_A = adding_mask(datapreprocess(entries[1]))
            sentences_A.append(sentence_A)
            masks_A.append(mask_A)
            sentence_B, mask_B = adding_mask(datapreprocess(entries[2]))
            sentences_B.append(sentence_B)
            masks_B.append(mask_B)
            relatedness.append(float(entries[3]))
    return np.array(sentences_A), np.array(sentences_B), np.array(relatedness), np.array(masks_A), np.array(masks_B)

def load_wordembedding(embeddings_path='./glove.6B.50d.txt', emb_dim=50):
    
    embeddings = [[.0]*emb_dim]
    word2id = {'<unknown>': 0} 
    id2word = {0: '<unknown>'}
    
    with open(embeddings_path, 'r', encoding='utf-8') as file:
        for i, embedding in enumerate(file):
            embedding = embedding.rstrip('\n').split(' ')
            embeddings.append([float(emb) for emb in embedding[1:]])
            word2id[embedding[0]] = int(i+1)
            id2word[str(i+1)] = embedding[0]
    return np.array(embeddings), word2id, id2word

def word_to_id(sentences, word2id):
    
    length = len(sentences)
    for i in range(length):
        for j, word in enumerate(sentences[i]):
            if not word in word2id:
                word = '<unknown>'
            sentences[i][j] = word2id[word]
            
    return sentences

def id_to_word(sentences, id2word):
    for i in range(len(sentences)):
        for j in range(len(sentences[i])):
            sentences[i][j] = id2word[sentences[i][j]]
    return sentences




def next_batch(data, batch_size):
    
    
    sentences_A, sentences_B, y_train, masks_A, masks_B = data
    size = len(sentences_A)
    indexes = np.arange(0, size)
    np.random.shuffle(indexes)
    
    sentences_A = [sentences_A[i] for i in indexes]
    sentences_B = [sentences_B[i] for i in indexes]
    masks_A = [masks_A[i] for i in indexes]
    masks_B = [masks_B[i] for i in indexes]
    y_train = [y_train[i] for i in indexes]
    
    
    nbatch = math.ceil(size / batch_size)
    for i in range(nbatch):
        offset = i * batch_size
        sentence_A = sentences_A[offset: offset + batch_size]
        sentence_B = sentences_B[offset: offset + batch_size]
        mask_A = masks_A[offset: offset + batch_size]
        mask_B = masks_B[offset: offset + batch_size]
        relatedness = y_train[offset: offset + batch_size]
        
        yield [sentence_A, sentence_B, relatedness, mask_A, mask_B]

    
    
def run_epoch(model, session, data, global_steps):
    
    for step, (sentence_A, sentence_B, relatedness, mask_A, mask_B) in enumerate(next_batch(data, batch_size= 32)):
        feed_dict = {
            model.sentence_A: sentence_A,
            model.sentence_B: sentence_B,
            model.mask_A: mask_A,
            model.mask_B: mask_B,
            model.relatedness: relatedness
        }
        
        fetches = [model.loss, model.similarity, model.train_op]
        loss, similarity, _ = session.run(fetches, feed_dict)
        
        if global_steps % 100 == 0:
            print('Step {}, Loss: {}'.format(global_steps, loss))
        
        global_steps += 1
        
    return global_steps
        
        

def train():
    with tf.Session() as session:
        
        sentences_A, sentences_B, relatedness, masks_A, masks_B = load_dataset()
        
        embeddings, word2id, id2word = load_wordembedding(emb_dim=300)
        
        sentences_A = word_to_id(sentences_A, word2id)
        sentences_B = word_to_id(sentences_B, word2id)
        
        data = [sentences_A, sentences_B, relatedness, masks_A, masks_B]
        
        initializer = tf.random_normal_initializer(0.0, 0.2, dtype=tf.float32)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            model = SiameseLSTM(batch_size = 32,learning_rate= 0.0001,emb_dim= 50,hidden_size= 50,max_len=100,num_epoch= 360,max_grad_norm= 5,vocab_size= 400001, sess=session, embeddings=embeddings, is_training=True)
        init = tf.global_variables_initializer()
        session.run(init)
        
        global_steps = 0
        for i in range(300):
            global_steps = run_epoch(model, session, data, global_steps)
        
            
train()
