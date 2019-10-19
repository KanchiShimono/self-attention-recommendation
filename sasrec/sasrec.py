from typing import Tuple

from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras_pos_embd import PositionEmbedding
from keras_position_wise_feed_forward import FeedForward
from tensorflow import Tensor
from tensorflow.keras.layers import Add, Dropout, Embedding, Input, Masking
from tensorflow.keras.models import Model


def build_model(max_len: int,
                input_dim: int,
                embedding_dim: int,
                feed_forward_units: int,
                head_num=1,
                block_num=1,
                dropout_rate=0.5) -> Tuple[Model, Embedding]:

    inputs = Input(shape=(max_len))
    emb = Embedding(input_dim=input_dim,
                    output_dim=embedding_dim,
                    mask_zero=True)
    x = emb(inputs)
    pos_emb = PositionEmbedding(
        input_dim=max_len,
        output_dim=embedding_dim,
        mode=PositionEmbedding.MODE_ADD,
        mask_zero=True
    )(x)
    y = Dropout(dropout_rate)(pos_emb)

    for _ in range(block_num):
        y = block(y, head_num, feed_forward_units, dropout_rate)

    model = Model(inputs=inputs, outputs=y)

    return model, emb


def block(attention_input,
          head_num: int,
          feed_forward_units: int,
          dropout_rate: float) -> Tensor:

    attention_x = MultiHeadAttention(
            head_num=head_num,
            activation=None,
            use_bias=False,
            history_only=True,
            trainable=True,
    )(attention_input)
    attention_x = Dropout(dropout_rate)(attention_x)
    attention_x = Add()([attention_input, attention_x])
    feed_forward_input = LayerNormalization(trainable=True)(attention_x)

    feed_forward_x = FeedForward(
            units=feed_forward_units,
            activation='relu',
            trainable=True
    )(feed_forward_input)
    feed_forward_x = Dropout(dropout_rate)(feed_forward_x)
    feed_forward_x = Add()([feed_forward_input, feed_forward_x])
    block_output = LayerNormalization(trainable=True)(feed_forward_x)

    return block_output
