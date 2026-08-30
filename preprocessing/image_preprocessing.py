import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess MRI image

    Steps:
    1. Read Image
    2. Convert BGR → RGB
    3. Gaussian Blur
    4. CLAHE on L channel
    5. Resize to 224×224
    6. Normalize
    """

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((l, a, b))

    # Convert back to RGB
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize
    image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
    return image