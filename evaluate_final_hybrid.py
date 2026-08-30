import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import label_binarize

from preprocessing.image_preprocessing import preprocess_image
from models.hybrid_model import build_hybrid_model

# =====================================================
# SETTINGS
# =====================================================

BATCH_SIZE = 8
NUM_CLASSES = 4
AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# FOCAL LOSS
# =====================================================

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

# =====================================================
# LOAD TEST CSV
# =====================================================

test_df = pd.read_csv("test.csv")

print("Number of Test Images :", len(test_df))

# =====================================================
# LABEL ENCODING
# =====================================================

class_names = sorted(test_df["Label"].unique())

label_to_index = {
    name: idx
    for idx, name in enumerate(class_names)
}

print("\nClass Mapping")

for k, v in label_to_index.items():
    print(k, "->", v)


# =====================================================
# IMAGE LOADER
# =====================================================

def load_image(path, label):

    image = preprocess_image(
        path.numpy().decode()
    )

    label = label_to_index[
        label.numpy().decode()
    ]

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

# =====================================================
# BUILD TEST DATASET
# =====================================================

test_dataset = tf.data.Dataset.from_tensor_slices(

    (
        test_df["Path"].values,
        test_df["Label"].values
    )

)

test_dataset = (

    test_dataset
    .map(
        tf_load_image,
        num_parallel_calls=AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)

)

# =====================================================
# BUILD MODEL
# =====================================================


model, _, _ = build_hybrid_model(
    input_shape=(224,224,3),
    num_classes=NUM_CLASSES
)

model.load_weights("best_hybrid_cbam_cleaned.weights.h5")
print("\nBest model loaded successfully.")

model.compile(
    optimizer="adam",
    loss=focal_loss(gamma=2.0, alpha=0.5),
    metrics=["accuracy"]
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.preprocessing import label_binarize

loss, accuracy = model.evaluate(test_dataset, verbose=0)
import gc
gc.collect()

print("TEST RESULTS")
print("="*60)
print(f"Loss      : {loss:.4f}")
print(f"Accuracy  : {accuracy*100:.2f}%")

# PREDICTIONS
# =====================================================

predictions = []

y_true = []

for images, labels in test_dataset:

    preds = model.predict(images, verbose=0)

    predictions.append(preds)

    y_true.extend(labels.numpy())

predictions = np.concatenate(predictions, axis=0)

y_true = np.array(y_true)

y_pred = np.argmax(predictions, axis=1)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\nPrecision :", round(precision,4))
print("Recall    :", round(recall,4))
print("F1 Score  :", round(f1,4))

balanced_acc = balanced_accuracy_score(y_true, y_pred)

print(f"Balanced Accuracy : {balanced_acc*100:.2f}%")
balanced_acc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Balanced Accuracy : {balanced_acc:.4f}")
print(f"Cohen's Kappa    : {kappa:.4f}")
print(f"Matthews CC      : {mcc:.4f}")

print("\n" + "="*60)
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

with open("classification_report.txt","w") as f:
    f.write(report)

cm = confusion_matrix(y_true,y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

y_true_bin = label_binarize(
    y_true,
    classes=[0,1,2,3]
)

plt.figure(figsize=(8,6))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:,i],
        predictions[:,i]
    )

    roc_auc = auc(fpr,tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} (AUC={roc_auc:.3f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

np.save(
    "prediction_probabilities.npy",
    predictions
)

np.save(
    "true_labels.npy",
    y_true
)

with open("results.txt","w") as f:

    f.write(f"Accuracy : {accuracy*100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
print("\nEvaluation Completed Successfully.")