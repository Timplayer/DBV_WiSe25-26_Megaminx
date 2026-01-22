
import cv2
import math
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from color_mapping import get_model, classify, load_color_samples
from src.canonical import FACE_NEIGHBORS

MAX_FACES = 3
CENTER_RADIUS = 6
MAX_ANGLE_DIFF = ((math.pi / 3) + 0.1)

def show(title, img, max_w=1200, max_h=800):
    """Show an image (auto-resized) and wait for a key.
       Press any key to continue; ESC exits."""

    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        raise SystemExit

def select_face_centers(candidates, max_faces=MAX_FACES):
    # greedy non-max suppression on center distance so we keep distinct faces
    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = []
    for score, x, y, a, b, angle, rad in candidates:
        too_close = False
        for _, px, py, pa, pb, _, prad in selected:
            if math.hypot(x - px, y - py) < 0.8 * (rad + prad):
                too_close = True
                break
        if too_close:
            continue
        selected.append((score, x, y, a, b, angle, rad))
        if len(selected) == max_faces:
            break
    return selected

def find_face_centers(contours, minR, maxR, max_faces=MAX_FACES):
    """Return up to `max_faces` center candidates; allow ellipses (flattened circles)."""
    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 400:
            continue

        if len(c) < 5:
            continue  # need enough points to fit an ellipse

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        a, b = 0.5 * MA, 0.5 * ma
        rad = max(a,b)

        if rad < minR or rad > maxR:
            continue

        ellipse_area = math.pi * a * b
        fill = min(area / ellipse_area, ellipse_area / max(area, 1))
        aspect = min(a, b) / max(a, b)

        score = fill * 0.9 + aspect * 0.1
        print(f"Candidate {i}: center=({x:.1f},{y:.1f}) axes=({a:.1f},{b:.1f}) angle={angle:.1f} area={area:.1f} fill={fill:.3f} aspect={aspect:.3f} score={score:.3f}")
        candidates.append((score, int(x), int(y), a, b, angle, rad))

    return select_face_centers(candidates, max_faces=max_faces)


def sample_center_color(img, x, y, radius=CENTER_RADIUS):
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch_bgr = img[y0:y1, x0:x1]
    b_mean, g_mean, r_mean = patch_bgr.reshape(-1, 3).mean(axis=0)

    return r_mean, g_mean, b_mean

def find_sticker_for_face(face, contours, img):
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 50000 or area < 500:
            continue

        M = cv2.moments(contour)
        if M['m00'] != 0:
            sticker_x = int(M['m10'] / M['m00'])
            sticker_y = int(M['m01'] / M['m00'])

            face_x = face["center"][0]
            face_y = face["center"][1]

            distance = math.sqrt((sticker_x - face_x)**2 + (sticker_y - face_y)**2)

            if distance < face["center"][2] * 3.5 and distance > face["center"][2]:

                r_mean, g_mean, b_mean = sample_center_color(img, sticker_x, sticker_y)

                sticker = {
                    "center" : (sticker_x, sticker_y),
                    "color" : (r_mean, g_mean, b_mean),
                }

                face["sticker"].append(sticker)

def pre_process_img(img):
    edge_src = cv2.GaussianBlur(img, (7, 7), 0)

    sobel_x = cv2.Sobel(edge_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edge_src, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    _, edges = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small = [c for c in all_contours if cv2.contourArea(c) < 100]
    if small:
        cv2.drawContours(edges, small, -1, 0, thickness=cv2.FILLED)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)


    h, w = edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(edges, mask, (0, 0), 255)
    filled = cv2.bitwise_not(edges)

    thr, mask_clean = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_clean = cv2.bitwise_and(mask_clean, filled)

    return mask_clean



def find_index_of_stickers(faces):

    for face in faces:

        face_angle = []

        for other_face in faces:
            if not face is other_face:

                center_x, center_y, _ = face["center"]
                other_center_x, other_center_y, _ = other_face["center"]

                dx = other_center_x - center_x
                dy = other_center_y - center_y

                angle = math.atan2(dy, dx)

                entry = {
                    "label": other_face["label"],
                    "angle": angle,
                    "stickers" : []
                }

                face_angle.append(entry)

        stickers = face["sticker"]
        if not stickers:
            continue

        center_x, center_y, _ = face["center"]


        for sticker in stickers:
            sticker_x, sticker_y = sticker["center"]
            dx = sticker_x - center_x
            dy = sticker_y - center_y

            angle = math.atan2(dy, dx)

            sticker["angle"] = angle

            for entry in face_angle:
                angle_diff = (angle - entry["angle"]) % (2 * math.pi)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                if angle_diff < MAX_ANGLE_DIFF:
                    entry["stickers"].append(sticker)


        first_sticker = face_angle[0]["stickers"][0]
        for sticker in face_angle[0]["stickers"]:
            if ((sticker["angle"] - first_sticker["angle"]) % (2*math.pi)) < (MAX_ANGLE_DIFF * 2):
                first_sticker = sticker
        offset_angle = float(first_sticker["angle"]) # save offset angle so we can offset the angles of all stickers

        for sticker in stickers:
            sticker["angle"] = (sticker["angle"] - offset_angle) % (2 * math.pi)

        stickers.sort(key=lambda s: s["angle"])

        offset = FACE_NEIGHBORS[face["label"]].index(face_angle[0]["label"]) * 2

        stickers_ordered = []
        for i in range(len(stickers)):
            idx = (i + offset) % len(stickers)
            stickers_ordered.append(stickers[idx])

        face["sticker"] = stickers_ordered

    return None










def process_image(path):

    img0 = cv2.imread(str(path))
    assert img0 is not None, f"Image not found: {path}"

    h0, w0 = img0.shape[:2]
    scale = 2000.0 / max(w0, h0)
    img = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)), cv2.INTER_AREA) if scale < 1 else img0.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    processed = pre_process_img(V)

    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"{path.name}: No contours found—check lighting or thresholds.")
        return [], []

    H, W = V.shape
    minR, maxR = int(0.02 * min(H,W)), int(0.5 * min(H,W))

    centers = find_face_centers(contours, minR, maxR, max_faces=MAX_FACES)


    faces = []
    for score, x, y, a, b, angle, rad in centers:
        r_mean, g_mean, b_mean = sample_center_color(img, x, y)

        face = {
            "center": (x, y, rad),
            "color": (r_mean, g_mean, b_mean),
            "sticker": [],
            "label" :  classify((r_mean, g_mean, b_mean), model=get_model())
        }

        faces.append(face)


    for face in faces:
        find_sticker_for_face(face, contours, img)

        for sticker in face["sticker"]:
            sticker["label"] = classify(sticker["color"], model=get_model())

            cv2.circle(img, sticker["center"], 5, (0, 255, 0), thickness=-1)


    show(f"Processed {path.name}", img)

    find_index_of_stickers(faces)

    for face in faces:
        for i, sticker in enumerate(face["sticker"]):
            cv2.putText(img, f"{i}", sticker["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    show(f"Processed {path.name}", img)

    return face


IMAGE_DIR = Path("../data/test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    image_paths = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
    if not image_paths:
        raise SystemExit(f"No image files found in {IMAGE_DIR}")

    for path in image_paths:
        process_image(path)


if __name__ == "__main__":
    main()
