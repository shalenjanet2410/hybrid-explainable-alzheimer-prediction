import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB3
)
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    BatchNormalization,
    Dense,
    Dropout,
    Concatenate
)

from tensorflow.keras.models import Model

from models.cbam import cbam_block


def build_hybrid_model(
        input_shape=(224, 224, 3),
        num_classes=4):

    # =====================================================
    # INPUT
    # =====================================================

    inputs = Input(shape=input_shape)

    
   
    # =====================================================
    # RESNET50 BRANCH
    # =====================================================

    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    resnet.trainable = False

    resnet_features = resnet.output

    resnet_features = cbam_block(resnet_features)

    resnet_gap = GlobalAveragePooling2D(
        name="resnet_gap"
    )(resnet_features)

    resnet_gmp = GlobalMaxPooling2D(
        name="resnet_gmp"
    )(resnet_features)

    resnet_features = Concatenate()([
        resnet_gap,
        resnet_gmp
    ])

    # =====================================================
    # EFFICIENTNETB0 BRANCH
    # =====================================================

    efficientnet = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )

    efficientnet.trainable = False

    efficient_features = efficientnet.output

    efficient_features = cbam_block(
        efficient_features
    )

    efficient_gap = GlobalAveragePooling2D(
        name="efficientnet_gap"
    )(efficient_features)

    efficient_gmp = GlobalMaxPooling2D(
        name="efficientnet_gmp"
    )(efficient_features)

    efficient_features = Concatenate()([
        efficient_gap,
        efficient_gmp
    ])

    # =====================================================
    # FEATURE FUSION
    # =====================================================

    fusion = Concatenate(name="feature_fusion")([
        resnet_features,
        efficient_features
    ])

    fusion = Dense(
        512,
        activation="relu",
        kernel_regularizer=l2(1e-4),
        name="fusion_dense"
    )(fusion)

    fusion = BatchNormalization(
        name="fusion_bn"
    )(fusion)

    fusion = Dropout(
        0.5,
        name="fusion_dropout"
    )(fusion)


    # =====================================================
    # CLASSIFIER
    # =====================================================


    x = Dense(
        512,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(fusion)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(0.4)(x)

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(1e-4),
        name="predictions"
    )(x)
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Hybrid_ResNet50_EfficientNet_CBAM"
    )

    print("\n" + "="*60)
    print("Hybrid ResNet50 + EfficientNetB3 + CBAM")
    print("="*60)
    
    return model, resnet, efficientnet
