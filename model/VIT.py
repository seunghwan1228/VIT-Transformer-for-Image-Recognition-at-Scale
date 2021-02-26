import tensorflow as tf

from model.resnet import CNNRoot, ResidualUnit, ResidualUnitShirink, ResidualUnitResCNN
from model.transformer import TransformerDenseBlock, MLPHead


class VisionTransformer(tf.keras.Model):
    """
    This Models is not Hybrid architecture,
    requires to transform image to patch
    """

    def __init__(self,
                 image_size,
                 patch_size,
                 num_layers,
                 num_classes,
                 model_dim,
                 num_heads,
                 feed_forward_units,
                 dropout_rate,
                 mlp_size,
                 first_strides,
                 max_pos,
                 res_block=5,
                 **kwargs):
        """
        res_w/o shrink 2 - res w/ shrink 1
        res_w/o shrink 3 - res w/ shrink 1
        res_w/o shrink 4 <= extracted patches
        """
        super(VisionTransformer, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.feed_forward_units = feed_forward_units
        self.dropout_rate = dropout_rate
        self.mlp_size = mlp_size
        num_patches = (image_size // patch_size) ** 2  # image -> patchs: total number of patches == 14**2
        self.max_pos = max_pos
        self.first_strides = first_strides
        self.res_block = res_block
        self.res_num_layer = 9
        self.res_shrink_num_layer = 3  # 224 -> 14

        self.pos_emb = self.add_weight(name='pos_emb', shape=(1, max_pos, model_dim))
        self.cls_emb = self.add_weight(name='cls_emb', shape=(1, 1, model_dim))

        self.dense_proj = tf.keras.layers.Dense(model_dim)
        self.enc_layer = [TransformerDenseBlock(model_dim, num_heads, feed_forward_units, dropout_rate, i) for i in
                          range(num_layers)]
        self.mlp_head = MLPHead(mlp_size, num_classes, dropout_rate)
        self.patch_dim = (self.patch_size ** 2) * 3

        self.cnn_root = CNNRoot(filters=model_dim, kernel_size=7, strides=first_strides, pool_size=3, pool_strides=2,
                                padding='same')
        self.resnet_wo_shrink = [ResidualUnit(filters=model_dim, strides=1, padding='same') for _ in
                                 range(self.res_num_layer)]
        self.resnet_shrink = [ResidualUnitShirink(filters=model_dim, strides=1, padding='same') for _ in
                              range(self.res_shrink_num_layer)]
        self.resnet_w_shortcut = [ResidualUnitResCNN(filters=model_dim, strides=1, padding='same') for _ in
                                  range(res_block)]

    def extract_patches(self, images):
        """
        This is hybrid format,
        input the image, extract its patches
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, inputs):
        # inputs: [b, seq, dim]
        batch_size = tf.shape(inputs)[0]
        x = self.cnn_root(inputs)

        for res_b_idx in range(2):
            x = self.resnet_wo_shrink[res_b_idx](x)
        x = self.resnet_shrink[0](x)
        for res_b in range(2, 5):
            x = self.resnet_wo_shrink[res_b](x)
        x = self.resnet_shrink[1](x)
        for res_b in range(5, 9):
            x = self.resnet_wo_shrink[res_b](x)
        x = self.resnet_shrink[2](x)
        for res_b in self.resnet_w_shortcut:
            x = res_b(x)
        # x = self.extract_patches(x)  # remove for hybrid model
        x = tf.reshape(x, [batch_size, -1, self.model_dim])
        _seq = tf.shape(x)[1]
        x = self.dense_proj(x)  # [b, seq, dim]
        cls_emb = tf.broadcast_to(self.cls_emb,
                                  [batch_size, 1, self.model_dim])  # Broad cast [1, 1, dim] -> [B, 1, dim]
        x = tf.concat([cls_emb, x], axis=1)  # cls_emb:seq
        x = x + self.pos_emb[:, :_seq + 1, :]

        enc_input = x

        for enc_ in self.enc_layer:
            enc_input = enc_(enc_input)
        x = self.mlp_head(enc_input[:, 0])
        return x


if __name__ == "__main__":
    model = VisionTransformer(224, 16, 4, 2, 128, 4, 128, 0.1, 128, 1, 5000)
    sample_img = tf.random.uniform(shape=(1, 224, 224, 3))
    tmp = model(sample_img)
    model.summary()