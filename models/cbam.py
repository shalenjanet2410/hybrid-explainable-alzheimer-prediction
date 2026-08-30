import tensorflow as tf

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Reshape,
    Multiply,
    Add,
    Conv2D,
    Concatenate,
    Activation,
    Lambda
)


def cbam_block(feature_map, ratio=8):

    channels = feature_map.shape[-1]

    # =====================================================
    # Channel Attention
    # =====================================================

    avg_pool = GlobalAveragePooling2D()(feature_map)
    max_pool = GlobalMaxPooling2D()(feature_map)

    avg_pool = Reshape((1, 1, channels))(avg_pool)
    max_pool = Reshape((1, 1, channels))(max_pool)

    shared_dense_one = Dense(
        channels // ratio,
        activation="relu"
    )

    shared_dense_two = Dense(channels)

    avg_out = shared_dense_two(
        shared_dense_one(avg_pool)
    )

    max_out = shared_dense_two(
        shared_dense_one(max_pool)
    )

    channel_attention = Add()([
        avg_out,
        max_out
    ])

    channel_attention = Activation(
        "sigmoid"
    )(channel_attention)

    feature_map = Multiply()([
        feature_map,
        channel_attention
    ])

    # =====================================================
    # Spatial Attention
    # =====================================================

    avg_pool = Lambda(
        lambda x: tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )
    )(feature_map)

    max_pool = Lambda(
        lambda x: tf.reduce_max(
            x,
            axis=-1,
            keepdims=True
        )
    )(feature_map)

    concat = Concatenate(axis=-1)([
        avg_pool,
        max_pool
    ])

    spatial_attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    feature_map = Multiply()([
        feature_map,
        spatial_attention
    ])

    return feature_map
