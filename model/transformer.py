import tensorflow as tf
from model.transformer_utils import ScaledDotAttention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.depth = self.model_dim // self.num_heads
        assert (self.model_dim % self.num_heads) == 0, f'The [ Model_dim % Num_heads ] should be 0 not {self.depth}'
        self.key_dense = tf.keras.layers.Dense(model_dim)
        self.query_dense = tf.keras.layers.Dense(model_dim)
        self.value_dense = tf.keras.layers.Dense(model_dim)
        self.output_linear = tf.keras.layers.Dense(model_dim)
        self.attn_layer = ScaledDotAttention()

    def split_head(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))  # [b, seq, head, depth]
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [b, head, seq, depth]

    def merge_head(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [b, seq, head, depth]
        return tf.reshape(x, shape=(batch_size, -1, self.model_dim))

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        query = self.query_dense(q)
        key = self.key_dense(k)
        value = self.value_dense(v)
        split_query = self.split_head(query, batch_size)
        split_key = self.split_head(key, batch_size)
        split_value = self.split_head(value, batch_size)
        attn_logit, attn_weight = self.attn_layer(split_query, split_key, split_value, mask=None)
        merge_attn = self.merge_head(attn_logit, batch_size)
        linear = self.output_linear(merge_attn)
        return linear, attn_weight


class FeedForwardDense(tf.keras.layers.Layer):
    def __init__(self, feed_forward_units, model_dim, dropout_rate, **kwargs):
        super(FeedForwardDense, self).__init__(**kwargs)
        self.feed_forward_units = feed_forward_units
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.feed_forward_dense = tf.keras.layers.Dense(feed_forward_units)
        self.feed_foraward_dr = tf.keras.layers.Dropout(dropout_rate)
        self.output_linear = tf.keras.layers.Dense(model_dim)

    def call(self, inputs):
        x = self.feed_forward_dense(inputs)
        x = tf.nn.gelu(x)
        x = self.feed_foraward_dr(x)
        return self.output_linear(x)


class FeedForwardConv(tf.keras.layers.Layer):
    def __init_(self, feed_forward_units, model_dim, kernel_size, strides, padding, dropout_rate, **kwargs):
        super(FeedForwardConv, self).__init__(**kwargs)
        self.feed_forward_units = feed_forward_units
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dropout_rate = dropout_rate

        self.feed_forward_conv = tf.keras.layers.Conv2D(filters=feed_forward_units,
                                                        kernel_size=kernel_size,
                                                        stirdes=strides,
                                                        padding=padding)
        self.feed_forward_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_conv = tf.keras.layers.Conv2D(filters=model_dim,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding)

    def call(self, inputs):
        x = self.feed_forward_conv(inputs)
        x = tf.nn.gelu(x)
        x = self.feed_forward_dropout(x)
        return self.output_conv(x)


class TransformerDenseBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, feed_forward_units, dropout_rate, layer_idx, **kwargs):
        """
        model_dim: int
        num_heads: int
        feed_forward_units: int
        dropout_rate: float
        layer_idx: int
        """
        super(TransformerDenseBlock, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.feed_forward_units = feed_forward_units
        self.dropout_rate = dropout_rate
        self.prior_mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'prior_mha_norm_{layer_idx}')
        self.mha = MultiHeadAttention(self.model_dim, self.num_heads, name=f'mha_{layer_idx}')
        self.prior_mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'prior_mlp_norm_{layer_idx}')
        self.mlp = FeedForwardDense(self.feed_forward_units, self.model_dim, self.dropout_rate, name=f'mlp_{layer_idx}')

    def call(self, inputs):
        x = self.prior_mha_norm(inputs)
        attn_logits, _ = self.mha(x, x, x, mask=None)
        res = inputs+attn_logits
        x = self.prior_mlp_norm(res)
        x = self.mlp(x)
        return x + res


class TransformerConvBlock(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, feed_forward_units, dropout_rate, kernel_size, strides, padding, layer_idx, **kwargs):
        """
        model_dim: int
        num_heads: int
        feed_forward_units: int
        dropout_rate: float
        kernel_size: int
        strides: int
        padding: str
        layer_idx: int
        """
        super(TransformerConvBlock, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.feed_forward_units = feed_forward_units
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.prior_mha_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'prior_mha_norm_{layer_idx}')
        self.mha = MultiHeadAttention(self.model_dim, self.num_heads, name=f'mha_{layer_idx}')
        self.prior_mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f'prior_mlp_norm_{layer_idx}')
        self.mlp = FeedForwardConv(feed_forward_units = self.feed_forward_units,
                                   model_dim = self.model_dim,
                                   dropout_rate = self.dropout_rate,
                                   kernel_size = self.kernel_size,
                                   strides = self.strides,
                                   padding = self.padding,
                                   name=f'mlp_{layer_idx}')

    def call(self, inputs):
        x = self.prior_mha_norm(inputs)
        attn_logits, attn_weights = self.mha(x, x, x, mask=None)
        res = inputs+attn_logits
        x = self.prior_mlp_norm(res)
        x = self.mlp(x)
        return x + res


class MLPHead(tf.keras.layers.Layer):
    def __init__(self, mlp_size, num_classes, dropout_rate, **kwargs):
        super(MLPHead, self).__init__(**kwargs)
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.prior_mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp_inter = tf.keras.layers.Dense(mlp_size)
        self.mlp_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.output_linear = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.prior_mlp_norm(inputs)
        x = self.mlp_inter(x)
        x = tf.nn.gelu(x)
        x = self.mlp_dropout(x)
        return self.output_linear(x)