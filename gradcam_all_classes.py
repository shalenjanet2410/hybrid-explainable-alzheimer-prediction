import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocessing.image_preprocessing import preprocess_image
from models.hybrid_model import build_hybrid_model

# =====================================================
# SETTINGS
# =====================================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

LABEL_TO_INDEX = {
    "MildDemented": 0,
    "ModerateDemented": 1,
    "NonDemented": 2,
    "VeryMildDemented": 3
}

# =====================================================
# LOAD MODEL
# =====================================================

model, resnet, efficientnet = build_hybrid_model(
    input_shape=(224,224,3),
    num_classes=NUM_CLASSES
)

model.load_weights(
    "best_hybrid_cbam_cleaned.weights.h5"
)

print("="*60)
print("Model Loaded Successfully")
print("="*60)

# =====================================================
# LOAD TEST CSV
# =====================================================

test_df = pd.read_csv("test.csv")

print("Total Test Images :", len(test_df))

# =====================================================
# GRAD-CAM FUNCTION
# =====================================================

def make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()

# =====================================================
# FIND THE BEST CORRECTLY CLASSIFIED IMAGE
# FOR EACH CLASS
# =====================================================

best_images = {}

print("\nSearching entire test set...\n")

for _, row in test_df.iterrows():

    image_path = row["Path"]

    true_label = LABEL_TO_INDEX[row["Label"]]

    image = preprocess_image(image_path)

    input_image = np.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        input_image,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    confidence = float(
        prediction[0][predicted_class]
    )

    # Ignore wrong predictions
    if predicted_class != true_label:
        continue

    if predicted_class not in best_images:

        best_images[predicted_class] = (
            confidence,
            image_path
        )

    elif confidence > best_images[predicted_class][0]:

        best_images[predicted_class] = (
            confidence,
            image_path
        )

print("="*60)
print("Best Correct Predictions")
print("="*60)

for cls in sorted(best_images.keys()):

    conf, img = best_images[cls]

    print(
        f"{CLASS_NAMES[cls]} : "
        f"{conf:.2%}"
    )

print()

# =====================================================
# GENERATE GRAD-CAM FOR EACH CLASS
# =====================================================

for predicted_class in sorted(best_images.keys()):

    confidence, image_path = best_images[predicted_class]

    print("="*60)
    print("Generating Grad-CAM")
    print("Class      :", CLASS_NAMES[predicted_class])
    print("Confidence :", f"{confidence:.2%}")
    print("Image      :", image_path)
    print("="*60)

    # ------------------------------------------
    # Load image
    # ------------------------------------------

    image = preprocess_image(image_path)

    input_image = np.expand_dims(
        image,
        axis=0
    )

    # ------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------

    heatmap = make_gradcam_heatmap(
        input_image,
        model,
        "conv5_block3_out"
    )

    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )

    # ------------------------------------------
    # Read original MRI
    # ------------------------------------------

    original = cv2.imread(image_path)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    original = cv2.resize(
        original,
        (224,224)
    )

    # ------------------------------------------
    # Create Brain Mask
    # ------------------------------------------

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5,5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        20,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print("Brain mask not found.")
        continue

    largest = max(
        contours,
        key=cv2.contourArea
    )

    brain_mask = np.zeros_like(gray)

    cv2.drawContours(
        brain_mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    brain_mask = cv2.GaussianBlur(
        brain_mask,
        (11,11),
        0
    )

    brain_mask = brain_mask.astype(
        np.float32
    ) / 255.0

    # ------------------------------------------
    # Apply Mask
    # ------------------------------------------

    heatmap = heatmap * brain_mask

    heatmap = heatmap / (
        heatmap.max() + 1e-8
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    heatmap_color = (
        heatmap_color *
        brain_mask[..., np.newaxis]
    ).astype(np.uint8)

    overlay = cv2.addWeighted(
        original,
        0.70,
        heatmap_color,
        0.50,
        0
    )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    plt.figure(figsize=(18,6))

    plt.subplot(141)
    plt.imshow(original)
    plt.title("Original MRI")
    plt.axis("off")

    plt.subplot(142)
    plt.imshow(brain_mask, cmap="gray")
    plt.title("Brain Mask")
    plt.axis("off")

    plt.subplot(143)
    plt.imshow(heatmap, cmap="jet")
    plt.title("ResNet50 Grad-CAM")
    plt.axis("off")

    plt.subplot(144)
    plt.imshow(overlay)
    plt.title(
        f"{CLASS_NAMES[predicted_class]}\nConfidence: {confidence:.2%}"
    )
    plt.axis("off")

    plt.tight_layout()

    filename = f"GradCAM_{CLASS_NAMES[predicted_class]}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(f"✓ Saved : {filename}")

print("\n" + "="*70)
print("Grad-CAM generation completed successfully!")
print("Generated images:")

for cls in sorted(best_images.keys()):
    print(f"  • GradCAM_{CLASS_NAMES[cls]}.png")

print("="*70)
