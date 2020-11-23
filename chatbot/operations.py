import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    """A function to calculate attention weights, given 4 heads: query, key, value and mask

    Parameters
    ----------
    query : [type]
        Query to search throughout the keys
    key : [type]
        Key tensor for the query to search
    value : [type]
        Values to apply attention weights produced by key query ops
    mask : [type]
        mask zero for irrelevant tokens

    Returns
    -------
    [type]
        [description]
    """

    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output
