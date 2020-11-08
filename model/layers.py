import tensorflow as tf
from model.transformer_utils import scaled_dot_attention, positional_encoding, gelu


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    The MultiHeadAttention with/out drop head
    """
    def __init__(self, num_heads, model_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = self.model_dim // self.num_heads
        assert self.model_dim % self.num_heads == 0, f'The Model dim // Num_heads are not Zero. Model Dim: {self.model_dim}, Heads: {self.num_heads} '

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
        attention_context, attention_weight = scaled_dot_attention(q=q_split,
                                                                   k=k_split,
                                                                   v=v_split,
                                                                   mask=mask)
        attention_context = tf.transpose(attention_context, perm=[0, 2, 1, 3])
        attention_context = tf.reshape(attention_context, shape=(batch_size, -1, self.model_dim))
        output = self.linear(attention_context)
        return output, attention_weight


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, ffn_units, model_dim, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.feed_units = ffn_units
        self.model_dim = model_dim

        self.ffn_dense = tf.keras.layers.Dense(ffn_units)
        self.out_dense = tf.keras.layers.Dense(model_dim)

    def call(self, inputs):
        x = self.ffn_dense(inputs)
        x = gelu(x, approximate=False)
        x = self.out_dense(x)
        return gelu(x, approximate=False)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_units, dropout_rate, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, model_dim=model_dim)
        self.mha_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_do = tf.keras.layers.Dropout(dropout_rate)
        self.ffn = FeedForward(ffn_units=ffn_units, model_dim=model_dim)
        self.ffn_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_do = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, mask, training):
        input_res = x
        x = self.mha_ln(x)
        x, attn_weight = self.mha(x, x, x, mask)
        x = self.do(x, training=training)
        attn_out = input_res + x
        ffn_out = self.ffn_ln(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.ffn_do(ffn_out, training=training)
        ffn_out = ffn_out + attn_out
        return ffn_out, attn_weight

