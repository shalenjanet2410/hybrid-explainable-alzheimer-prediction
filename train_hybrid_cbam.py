import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from preprocessing.image_preprocessing import preprocess_image
import tensorflow.keras.backend as K
import gc
import tensorflow as tf

gc.collect()
tf.keras.backend.clear_session()
# =====================================
# SETTINGS
# =====================================

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 4

AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(42)

def focal_loss(gamma=2.0, alpha=0.5):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)

        y_true = tf.one_hot(y_true, depth=NUM_CLASSES)

        y_pred = tf.clip_by_value(
            y_pred,
            K.epsilon(),
            1.0 - K.epsilon()
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow(1 - y_pred, gamma)

        loss = weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss
# =====================================
# LOAD CSV FILES
# =====================================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

print("Training Images :", len(train_df))
print("Validation Images :", len(val_df))

# =====================================
# LABEL ENCODING
# =====================================

class_names = sorted(train_df["Label"].unique())

label_to_index = {
    name: idx
    for idx, name in enumerate(class_names)
}

print("\nClass Mapping")

for k, v in label_to_index.items():
    print(k, "->", v)

# =====================================
# CLASS WEIGHTS
# =====================================

train_labels = train_df["Label"].map(label_to_index).values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {
    i: weights[i]
    for i in range(len(weights))
}

print("\nClass Weights")
print(class_weights)

def load_image(path, label):

    image = preprocess_image(path.numpy().decode())

    label = label_to_index[label.numpy().decode()]

    return image.astype(np.float32), np.int32(label)


def tf_load_image(path, label):

    image, label = tf.py_function(
        load_image,
        [path, label],
        [tf.float32, tf.int32]
    )

    image.set_shape((224,224,3))
    label.set_shape(())

    return image, label

# =====================================
# BUILD DATASET
# =====================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["Path"].values,
        train_df["Label"].values
    )
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (
        val_df["Path"].values,
        val_df["Label"].values
    )
)

train_dataset = (
    train_dataset
    .shuffle(len(train_df), seed=42)
    .map(tf_load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

val_dataset = (
    val_dataset
    .map(tf_load_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(1)
)

print("\nTensorFlow Dataset Created Successfully!")


# =====================================
# BUILD HYBRID MODEL

# =====================================
# BUILD HYBRID MODEL
# =====================================

from models.hybrid_model import build_hybrid_model

model, resnet, efficientnet = build_hybrid_model(
    input_shape=(224,224,3),
    num_classes=NUM_CLASSES
)

print("\nHybrid Model Created Successfully!")


# =====================================

# COMPILE MODEL
# =====================================

model.compile(

    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-5,
        weight_decay=1e-5
    ),

    loss=focal_loss(
        gamma=2.0,
        alpha=0.5
    ),

    metrics=[
        "accuracy"
    ]

)

print("\nModel Compiled Successfully!")

model.summary()

# =====================================
# CALLBACKS
# =====================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True

)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7

)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_hybrid_cbam_cleaned.weights.h5",
    
    monitor="val_loss",

    mode="min",

    save_best_only=True,

    save_weights_only=True,

    verbose=1

)

# =====================================
# STAGE 1 TRAINING
# =====================================

print("\n")
print("=" * 60)
print("Stage 1 : Training Classifier")
print("=" * 60)

history1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=15,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]

)

# =====================================
# STAGE 2 : FINE TUNING
# =====================================

print("\n")
print("=" * 60)
print("Stage 2 : Fine Tuning")
print("=" * 60)

# Unfreeze pretrained models
resnet.trainable = True
efficientnet.trainable = True

# Freeze early ResNet layers
for layer in resnet.layers[:-80]:
    layer.trainable = False

# Freeze early EfficientNet layers
for layer in efficientnet.layers[:-120]:
    layer.trainable = False

model.compile(

    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=5e-6,
        weight_decay=1e-5
    ),

    loss=focal_loss(
        gamma=2.0,
        alpha=0.5
    ),

    metrics=[
        "accuracy"
    ]
)
print("\nTrainable layers:")

count = np.sum([layer.trainable for layer in model.layers])

print(count)

history2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=30,

    class_weight=class_weights,

    initial_epoch=history1.epoch[-1] + 1,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]

)

history = {}

for key in history1.history.keys():

    history[key] = (
        history1.history[key] +
        history2.history[key]
    )
import json

with open("training_history.json", "w") as f:
    json.dump(history, f)

print("Training history saved successfully!")

# =====================================
# PLOT ACCURACY
# =====================================


plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_hybrid_cbam.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8,5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_hybrid_cbam.png", dpi=300, bbox_inches="tight")
plt.close()

model.save_weights("final_hybrid_cbam_cleaned.weights.h5")
model.save("final_hybrid_cbam_cleaned.keras")

print("\n")
print("="*60)
print("Training Completed Successfully!")
print("="*60)

