import tensorflow as tf
from model.transformer_utils import scaled_dot_attention, positional_encoding


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    The MultiHeadAttention with/out drop head
    '''
    def __init__(self, num_heads, model_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_heads
        assert self.model_dim % self.num_heads == 0, f'The Model dim // Num_heads are not Zero. Model Dim: {self.model_dim}, Heads: {self.num_heads}'

        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.linear = tf.keras.layers.Dense(model_dim)


    def split_head(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q_ = self.wq(q)
        k_ = self.wk(k)
        v_ = self.wv(v)
        q_split = self.split_head(q_, batch_size)
        k_split = self.split_head(k_, batch_size)
        v_split = self.split_head(v_, batch_size)
        attention_context, attention_weight = scaled_dot_attention(q = q_split,
                                                                   k = k_split,
                                                                   v = v_split,
                                                                   mask = mask)
        attention_context = tf.transpose(attention_context, perm=[0, 2, 1, 3])
        attention_context = tf.reshape(attention_context, shape=(batch_size, -1, self.model_dim))
        output = self.linear(attention_context)
        return output, attention_weight

