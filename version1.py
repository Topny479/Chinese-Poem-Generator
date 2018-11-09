# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import time

filename = './/newpoem.txt'

batch_size = 20
num_steps = 26        
lstm_size = 256       
num_layers = 2          
L_rate = 0.001
keep_prob = 0.5

# 生成字典并且编号
with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i,c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

def get_batches(arr, batch_size, n_steps):

    characters_per_batch = batch_size * n_steps
    n_batches = len(arr)//characters_per_batch
    
    arr = arr[:characters_per_batch*n_batches]
    
    arr = np.reshape(arr,[batch_size,-1])
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y_temp = arr[:,n+1:n+n_steps+1]
        #这些是废话吗,y_temp直接赋值给y不可以吗
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :y_temp.shape[1]] = y_temp
        
        yield x, y
        
def build_inputs(batch_size, num_steps):

    inputs = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name= 'targets')
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    #防止过拟合
    def build_cell(lstm_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    return cell, init_state


def  build_output (lstm_output, in_size, out_size):

    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=1.0))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):

    y_one_hot = tf.one_hot(targets,num_classes)
    y_shaped = tf.reshape(y_one_hot, [-1, num_classes])
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_shaped, logits=logits)
    loss = tf.reduce_mean(loss, name='loss')
    
    return loss


def  build_optimizer(loss, learning_rate, grad_clip):

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

class CharRNN:
    def __init__(self, num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=256, num_layers=2, learning_rate=L_rate,
                grad_clip=5, sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        
        tf.reset_default_graph()
        
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot,initial_state=self.initial_state)
        self.final_state = state
        
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime):
    samples = [c for c in prime]
    model = CharRNN(num_classes=len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)

epochs = 20

print_every_n = 50
checkpoint_path = './models/train.ckpt'
save_every_n = 200
model = CharRNN(num_classes=len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=256, num_layers=2, 
                learning_rate=L_rate)

saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #每一次训练都run一次
    save_path = saver.save(sess, checkpoint_path, global_step=1)

    saver.restore(sess, checkpoint_path+'-'+str(1))
    print("Model restored.")
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            if (counter % print_every_n == 0):
                end = time.time()
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "./models/train.ckpt")
                
    saver.save(sess, "./models/train.ckpt")
    
tf.train.get_checkpoint_state('models')

tf.train.latest_checkpoint('models')

checkpoint = tf.train.latest_checkpoint('models')
samp = sample(checkpoint, 28, 256, len(vocab), prime="春")
print(samp)