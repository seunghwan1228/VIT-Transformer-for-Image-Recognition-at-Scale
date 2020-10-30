import tensorflow as tf
import numpy as np


def get_angles(pos, i, model_dim):
    angle_rates = 1. / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    '''
    get_angles(pos, i, model_dim)

    pos: [[1],
          [2],
          [3],
          [4],
          [5]]

    i: [[1],[2],[3],[4],[5]]

    pos * i
      [[ 1,  2,  3,  4,  5],
       [ 2,  4,  6,  8, 10],
       [ 3,  6,  9, 12, 15],
       [ 4,  8, 12, 16, 20],
       [ 5, 10, 15, 20, 25]]
    '''
    angle_rates = get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(model_dim)[np.newaxis, :],
                             model_dim)
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])

    pos_emb = angle_rates[np.newaxis, ...]
    return tf.cast(pos_emb, tf.float32)


def scaled_dot_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    attn_score = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        attn_score += (mask * -1e9)

    attn_weights = tf.nn.softmax(attn_score, axis=-1)
    context_vector = tf.matmul(attn_weights, v)
    return context_vector, attn_weights

