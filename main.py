"""
Fun little afternoon project to use webcam and determine what cat meme face I'm doing.
"""

import cv2
import os
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

#Setup

#Huggingface model for face expressions classification
MODEL        = "mo-thecreator/vit-Facial-Expression-Recognition"
INFERENCE_INTERVAL = 3

#Params for OpenCV face detector to find face to give model
FACE_SCALE      = 1.1
FACE_NEIGHBORS  = 5
MIN_FACE_SIZE   = (60, 60)

#GUI/Display things
IMAGE_DIR = "cats"   
PANEL_WIDTH     = 300                   
EXPR_IMG_SIZE   = 220

# Expression labels — also the expected image filenames (without extension)
EXPRESSION_LABELS = ["happy", "sad", "anger", "surprise", "fear", "disgust", "neutral"] #these match the output labels of the model
LABEL_COLOURS = {
    "happy":    (0,   220, 80),
    "sad":      (200, 80,  0),
    "anger":    (0,   0,   220),
    "surprise": (0,   200, 220),
    "fear":     (150, 0,   200),
    "disgust":  (0,   150, 100),
    "neutral":  (180, 180, 180),
}
DEFAULT_COLOUR = (200, 200, 200)
EXTS = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]

print("Loading model from HuggingFace")

DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

expr_classifier = pipeline(
    "image-classification",
    model=MODEL,
    device=DEVICE,
    torch_dtype=DTYPE,
)

expr_classifier.model.eval()
if torch.cuda.is_available():
    expr_classifier.model.half()

print(f"Model loaded: device={'GPU' if DEVICE == 0 else 'CPU'} | dtype={DTYPE}")

#OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def load_expression_images(folder: str) -> dict:
    """Loads images for display

    Args:
        folder (str): parent dir of images

    Returns:
        dict: (expression, image)
    """
    images = {}

    if not os.path.isdir(folder):
        print(f"{folder} not found")
        return {label: None for label in EXPRESSION_LABELS}

    available = {}
    for file in os.listdir(folder):
        name, ext = os.path.splitext(file)
        if ext.lower() in EXTS:
            available[name.lower()] = os.path.join(folder, file)

    for label in EXPRESSION_LABELS:
        path = available.get(label)
        if path is None:
            print(f"No image for {label} in {folder}")
            images[label] = None
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Could not read {path}")
                images[label] = None
            else:
                images[label] = img
                print(f"Read {label}: {os.path.basename(path)}")
    return images

def resize_expr_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resizes the expression display image to be square, preserves aspect ratio w/ black borders

    Args:
        img (np.ndarray): expr image
        size (int): desired side len

    Returns:
        np.ndarray: resized image
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    background = np.zeros((size, size, 3), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2

    background[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return background

def predict_expression(face: np.ndarray) -> tuple[str, float]:
    """Runs expressions classification inference

    Args:
        face (np.ndarray): face image

    Returns:
        tuple: (label, score)
    """
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(face)
    with torch.no_grad():
        results = expr_classifier(pil_img)
    top   = results[0]
    label = top["label"].lower()
    score = top["score"]
    return label, score

def draw_results(image: np.ndarray, x: int, y: int, w: int, h: int, label: str, score: float):
    """Draws the results of det/class on display image

    Args:
        image (np.ndarray): image
        x (int): x
        y (int): y
        w (int): width
        h (int): height
        label (str): class label
        score (float): conf score
    """
    colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
    cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)

    text = f"{label}: {score*100:.0f}%"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 6
    rx1, ry1 = x, max(0, y - th - pad * 2 - baseline)
    rx2, ry2 = x + tw + pad * 2, y

    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), colour, -1)
    cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 0, 0), 1)
    cv2.putText(
        image, text,
        (rx1 + pad, ry2 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA,
    )

def confidence_bar(image: np.ndarray, x: int, y: int, w: int, score: float, colour: tuple):
    """Helper func to draw a confidence bar

    Args:
        image (np.ndarray): image
        x (int): x
        y (int): y
        w (int): width
        score (float): conf score
        colour (tuple): color
    """
    bar_h  = 6
    filled = int(w * score)
    cv2.rectangle(image, (x, y + 2), (x + w, y + 2 + bar_h), (50, 50, 50), -1)
    cv2.rectangle(image, (x, y + 2), (x + filled, y + 2 + bar_h), colour, -1)


def expr_display(height: int, panel_w: int, label: str, score: float,
                     expr_images: dict) -> np.ndarray:
    """Creates a side display panel with the expression image

    Args:
        height (int): height
        panel_w (int): width of panel
        label (str): class label
        score (float): conf score
        expr_images (dict): (expressions, images)

    Returns:
        np.ndarray: result expression image
    """
    panel  = np.full((height, panel_w, 3), 30, dtype=np.uint8)
    colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR) if label else DEFAULT_COLOUR

    cv2.rectangle(panel, (0, 0), (3, height), colour, -1)

    if label is None:
        msg = "Waiting…"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(panel, msg, ((panel_w - tw) // 2, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2, cv2.LINE_AA)
        return panel

    img = expr_images.get(label)
    y_off = 20

    if img is not None:
        img_sq = resize_expr_image(img, EXPR_IMG_SIZE)
        img_h, img_w = img_sq.shape[:2]
        x_off = (panel_w - img_w) // 2
        panel[y_off:y_off + img_h, x_off:x_off + img_w] = img_sq
        text_y = y_off + img_h + 32
    else:
        # Placeholder box when image is missing
        cv2.rectangle(panel, (20, y_off), (panel_w - 20, y_off + EXPR_IMG_SIZE),
                      (60, 60, 60), -1)
        no_img = "no image"
        (tw, _), _ = cv2.getTextSize(no_img, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(panel, no_img,
                    ((panel_w - tw) // 2, y_off + EXPR_IMG_SIZE // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        text_y = y_off + EXPR_IMG_SIZE + 32

    # Label name
    label_text = label.capitalize()
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(panel, label_text, ((panel_w - tw) // 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)

    # Confidence percentage
    conf_text = f"{score * 100:.1f}%"
    (tw2, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(panel, conf_text, ((panel_w - tw2) // 2, text_y + th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2, cv2.LINE_AA)

    # Confidence fill bar
    bar_x = 20
    bar_y = text_y + th + 28
    bar_w = panel_w - 40
    filled = int(bar_w * score)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (60, 60, 60), -1)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled, bar_y + 10), colour, -1)

    return panel

def main():
    expr_images = load_expression_images(IMAGE_DIR)
    loaded = sum(v is not None for v in expr_images.values())
    print(f"{loaded}/{len(EXPRESSION_LABELS)} expression images loaded")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Try differencet webcam source")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_idx  = 0
    face_cache = []   # list of (x, y, w, h, label, score)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE,
            minNeighbors=FACE_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
        )

        if frame_idx % INFERENCE_INTERVAL == 0 and len(faces):
            new_cache = []
            for (x, y, w, h) in faces:
                pad  = int(0.10 * min(w, h))
                fx   = max(0, x - pad)
                fy   = max(0, y - pad)
                fw   = min(frame.shape[1] - fx, w + 2 * pad)
                fh   = min(frame.shape[0] - fy, h + 2 * pad)
                crop = frame[fy:fy + fh, fx:fx + fw]
                if crop.size == 0:
                    continue
                label, score = predict_expression(crop)
                new_cache.append((x, y, w, h, label, score))
                # print(f"Face @ {x},{y}: {label} | ({score*100:.1f}%)")
            face_cache = new_cache

        # Draw results on webcam image
        for (x, y, w, h, label, score) in face_cache:
            colour = LABEL_COLOURS.get(label, DEFAULT_COLOUR)
            draw_results(frame, x, y, w, h, label, score)
            confidence_bar(frame, x, y + h + 2, w, score, colour)

        # Display cat expression, use the highest scoring face
        if face_cache:
            top_label, top_score = face_cache[0][4], face_cache[0][5]
        else:
            top_label, top_score = None, 0.0

        panel = expr_display(
            height=frame.shape[0],
            panel_w=PANEL_WIDTH,
            label=top_label,
            score=top_score,
            expr_images=expr_images,
        )

        combined = np.hstack([frame, panel])
        cv2.imshow("Expression Recognition", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()