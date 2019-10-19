import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.python.training.adam import AdamOptimizer

from sampler import WarpSampler
from sasrec import build_model
from util import data_partition, evaluate, evaluate_valid


@tf.function
def train_one_step(model, emb, data_x, pos, neg):
    logits = model(data_x, training=True)
    pos_emb = emb(pos)
    neg_emb = emb(neg)

    pos_logits = tf.reduce_sum(logits * pos_emb, axis=-1)
    neg_logits = tf.reduce_sum(logits * neg_emb, axis=-1)
    numeric_mask = tf.cast(model.compute_mask(data_x, None), tf.float32)

    loss = tf.reduce_sum(
        (
            -tf.math.log(tf.sigmoid(pos_logits) + K.epsilon())
            - tf.math.log(1. - tf.sigmoid(neg_logits) + K.epsilon())
        ) * numeric_mask
    ) / tf.reduce_sum(numeric_mask)

    return loss


if not os.getenv('TF_KERAS'):
    print('TF_KERAS is not set')
    sys.exit(1)


batch_size = 128
max_len = 20

dataset = data_partition('./data/movielens.txt')
# [user_train, user_valid, user_test, user_num, item_num] = dataset
[user_train, user_valid, user_test, user_num, item_num, item_count, cum_table] = dataset
num_batch = len(user_train) // (batch_size)

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: {:.2f}'.format(cc / len(user_train)))

sampler = WarpSampler(user_train,
                      user_num,
                      item_num,
                      cum_table,
                      batch_size=batch_size,
                      max_len=max_len,
                      n_workers=3)

model, emb = build_model(max_len=max_len,
                         input_dim=item_num + 1,
                         embedding_dim=50,
                         feed_forward_units=50,
                         head_num=1,
                         block_num=2,
                         dropout_rate=0.2)

optimizer = AdamOptimizer(0.001)
tbcb = TensorBoard(log_dir='/logs', histogram_freq=1, write_graph=True, write_grads=True, write_images=True, embeddings_freq=1)

loss_history = []
cos_loss_history = []

T = 0.0
t0 = time.time()

tbcb.set_model(model)
tbcb.on_train_begin()


f = open('log.txt', 'w')

try:
    for epoch in range(200):
        tbcb.on_epoch_begin(epoch)

        cos_loss = CosineSimilarity()

        for step in (range(num_batch)):
            tbcb.on_train_batch_begin(step)
            print('========== step: {:03d} / {:03d} ============\r'.format(step, num_batch), end='')

            u, seq, pos, neg = sampler.next_batch()
            seq = tf.convert_to_tensor(seq)
            pos = tf.convert_to_tensor(pos)
            neg = tf.convert_to_tensor(neg)

            with tf.GradientTape() as tape:
                loss = train_one_step(model, emb, seq, pos, neg)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            cos_loss(model(seq), emb(pos))
            tbcb.on_train_batch_end(step)

        loss_history.append(loss.numpy())
        cos_loss_history.append(cos_loss.result())
        # print('========== loss: {:03f} ============'.format(loss_history[-1]))
        print(
            'Epoch: {:03d}, Loss: {:.3f}, CosineSimilarity: {:.3f}'.format(
                epoch,
                loss_history[-1],
                cos_loss.result()
            )
        )
        logs = {
            'train_loss': loss,
            'cosine_similarity': cos_loss.result()
        }
        tbcb.on_epoch_end(epoch, logs=logs)

        if epoch % 5 == 0:
            t1 = time.time() - t0
            T += t1
            print('========== Evaluating ==========')
            t_test = evaluate(model, emb, dataset, max_len)
            t_valid = evaluate_valid(model, emb, dataset, max_len)
            print('Epoch: {:03d}, Time: {:f}, valid (NDCG@10: {:.4f}, HR@10: {:.4f}), test (NDCG@10: {:.4f}, HR@10: {:.4f})'.format(
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]
            ))
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()

        # if np.array(loss_history)[::-1].argsort().argsort()[0] > 3:
        if epoch - np.array(loss_history).argsort()[0] > 10:
            break

    tbcb.on_train_end()

except Exception as e:
    print(e)
    tbcb.on_train_end()
    f.close()
    sampler.close()

f.close()
sampler.close()
