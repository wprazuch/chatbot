import tensorflow as tf
from .operations import scaled_dot_product_attention


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head Attention Layer for Transformer network
    """

    def __init__(self, d_model, num_heads, name='multi_head_attention'):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads

        # depth per head * num_heads
        self.d_model = d_model

        # if cant be divided without rest, we cannot split equally between all heads
        assert d_model % self.num_heads == 0

        # depth per head for multi head attention
        self.depth = d_model // self.num_heads

        # query, key, value matrices for all heads - each having d_model dimensionality
        # to get Q, K, V for only one matrix, we will reshape these matrices (num_heads, d_model/num_heads)
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # Final Dense layer to multiply concatenated heads with
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        """splits input into multiple heads for attention operation by reshaping matrices
        Operation will be performed on Q, K, V matrices

        Parameters
        ----------
        inputs : [type]
            Input for the network
        batch_size : [type]
            batch size of the network

        Returns
        -------
        [type]
            heads for attention module
        """

        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))

        # Rearrange dimensions, batch still as first dim, then each head for attention, their dimensionality and dept
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : dict
            pass dictionary as an input, containing query, key, value and mask heads

        Returns
        -------
        [type]
            output from the Multi-head Attention layer
        """

        # get Q, K, V inputs from dictionary input
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']

        batch_size = tf.shape(query)[0]

        # pass heads through dense layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split query, key, value into multiple heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # perform Attention -> softmax((QK.T)/sqrt(dk) * V)
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # merge heads into single dimension to prepare for final matmul
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # perform final matmul
        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding class to add positional information to the input vector
    """

    def __init__(self, position: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model: int, num_heads: int, dropout: float, name: str = 'encoder_layer'):
    """Encoding layer for encoder -> encoder can 
    have multiple encoder layers linked sequentially

    Parameters
    ----------
    units : [type]
        [description]
    d_model : [type]
        [description]
    num_heads : [type]
        [description]
    dropout : [type]
        [description]
    name : str, optional
        [description], by default 'encoder_layer'

    Returns
    -------
    [type]
        [description]
    """

    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention = MultiHeadAttention(d_model, num_heads, name='attention')({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='encoder'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='encoder_layer_{}'.format(i)
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name='decoder_layer'):
    inputs = tf.keras.Input(shape=(None, d_model), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(d_model, num_heads, name='attention_1')({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })

    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name='attention_2')(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })

    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)

    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs, name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
                          outputs=outputs,
                          name=name)


def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name='transformer'):

    inputs = tf.keras.Input(shape=(None,), name='inputs')
    dec_inputs = tf.keras.Input(shape=(None,), name='dec_inputs')

    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, d_model=d_model,
        num_heads=num_heads, dropout=dropout)(
        inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size, num_layers=num_layers, units=units, d_model=d_model,
        num_heads=num_heads, dropout=dropout)(
        inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name='outputs')(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
