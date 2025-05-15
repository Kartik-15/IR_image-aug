
import cv2
import numpy as np
import os
import random

# === Settings ===
input_dir = r"C:\Users\ksing\Downloads\input_images"  # Update as needed
output_dir = r"C:\Users\ksing\Downloads\output_images"
os.makedirs(output_dir, exist_ok=True)

# === Tint variations (BGR format for OpenCV)
tints = {
    "warm": (0, 30, 80),
    "cool": (80, 30, 0),
}

# === Brightness levels
brightness_factors = {
    "normal": 1.2,
    "bright": 1.4
}

alpha = 0.25  # Tint blending strength

def apply_shadow(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    shadow_polygon = np.array([[0, h], [w, h], [w, int(0.6*h)], [0, int(0.7*h)]], dtype=np.int32)
    cv2.fillPoly(mask, [shadow_polygon], 100)
    shadow = cv2.merge([mask]*3)
    return cv2.addWeighted(image, 1, shadow, 0.4, 0)

def apply_reflection(image):
    overlay = image.copy()
    h, w = image.shape[:2]
    for i in range(0, h, 4):
        cv2.line(overlay, (0, i), (w, i), (255, 255, 255), 1)
    return cv2.addWeighted(image, 0.9, overlay, 0.1, 0)

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_random_occlusion(image):
    h, w = image.shape[:2]
    x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
    x2, y2 = random.randint(x1+20, w), random.randint(y1+20, h)
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    return image

def apply_perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([[random.randint(0,10),random.randint(0,10)],
                       [w-random.randint(0,10),random.randint(0,10)],
                       [random.randint(0,10),h-random.randint(0,10)],
                       [w-random.randint(0,10),h-random.randint(0,10)]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

# === Process images
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = os.path.splitext(filename)[0]

        for brightness_label, brightness in brightness_factors.items():
            bright_image = np.clip(image * brightness, 0, 255).astype(np.uint8)

            for tint_label, tint_color in tints.items():
                tint_layer = np.full_like(bright_image, tint_color, dtype=np.uint8)
                final_image = cv2.addWeighted(bright_image, 1 - alpha, tint_layer, alpha, 0)

                aug_image = final_image.copy()
                aug_image = apply_shadow(aug_image)
                aug_image = apply_reflection(aug_image)
                aug_image = apply_gaussian_blur(aug_image)
                aug_image = apply_random_occlusion(aug_image)
                aug_image = apply_perspective_transform(aug_image)

                output_filename = f"{base_name}_{brightness_label}_{tint_label}_aug.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

print(f"All augmented images saved to '{output_dir}' ✅")
