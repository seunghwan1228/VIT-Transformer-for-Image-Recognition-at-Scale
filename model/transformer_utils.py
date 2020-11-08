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


def gelu(features, approximate=False, name=None):

    """Compute the Gaussian Error Linear Unit (GELU) activation function.
    Gaussian error linear unit (GELU) computes
    `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
    The (GELU) nonlinearity weights inputs by their value, rather than gates
    inputs by their sign as in ReLU.
    For example:
    x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    y = tf.nn.gelu(x)
    y.numpy()
    array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
        dtype=float32)
    y = tf.nn.gelu(x, approximate=True)
    y.numpy()
    array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
        dtype=float32)
    Args:
      features: A `Tensor` representing preactivation values.
      approximate: An optional `bool`. Defaults to `False`. Whether to enable
        approximation.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` with the same type as `features`.
    References:
      [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).

      def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    features = tf.convert_to_tensor(features, name="features")
    if approximate:
        coeff = tf.math.cast(0.044715, features.dtype)
        return 0.5 * features * (
                1.0 + tf.math.tanh(0.7978845608028654 *
                                   (features + coeff * tf.math.pow(features, 3))))
    else:
        return 0.5 * features * (1.0 + tf.math.erf(
            features / tf.cast(1.4142135623730951, features.dtype)))