from visualize import visualize
from dataclasses import dataclass
from typing import Any

import cv2
import math
import numpy as np
from pathlib import Path

from color_mapping import get_model, classify
from src.canonical import FACE_NEIGHBORS

MAX_FACES = 3
CENTER_RADIUS = 6
MAX_ANGLE_DIFF = ((math.pi / 3) + 0.1)
DEBUG = True


def debug_show(title, img, max_w=1200, max_h=800):
    """Show an image (auto-resized) and wait for a key.
       Press any key to continue; ESC exits."""
    if not DEBUG:
        return

    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        raise SystemExit


@dataclass
class Sticker:
    center: tuple[int, int]
    color: tuple[int, int, int]  # rgb
    label: str
    angle: float | None
    dif_angle: float | None = None


@dataclass
class Face:
    center: tuple[float, float, float]
    color: tuple[int, int, int]  # rgb
    sticker: list[Sticker]
    label: str


@dataclass
class Entry:
    label: str
    angle: float
    sticker: list[Sticker]


def select_face_centers(candidates, max_faces=MAX_FACES):
    # greedy non-max suppression on center distance so we keep distinct faces
    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = []
    for score, x, y, a, b, angle, rad in candidates:
        too_close = False
        for _, px, py, pa, pb, _, p_rad in selected:
            if math.hypot(x - px, y - py) < 0.8 * (rad + p_rad):
                too_close = True
                break
        if too_close:
            continue
        selected.append((score, x, y, a, b, angle, rad))
        if len(selected) == max_faces:
            break
    return selected


def find_face_centers(contours: Any, min_radius: float, max_radius: float, img_rgb: cv2.typing.MatLike,
                      max_faces: int = MAX_FACES) -> list[Face]:
    """Return up to `max_faces` center candidates; allow ellipses (flattened circles)."""
    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 400:
            continue

        if len(c) < 5:
            continue  # need enough points to fit an ellipse

        (x, y), (ma_a, ma_b), angle = cv2.fitEllipse(c)
        a, b = 0.5 * ma_a, 0.5 * ma_b
        rad = max(a, b)

        if rad < min_radius or rad > max_radius:
            continue

        ellipse_area = math.pi * a * b
        fill = min(area / ellipse_area, ellipse_area / max(area, 1))
        aspect = min(a, b) / max(a, b)

        score = fill * 0.9 + aspect * 0.1
        print(
            f"Candidate {i}: center=({x:.1f},{y:.1f}) axes=({a:.1f},{b:.1f}) angle={angle:.1f} area={area:.1f} fill={fill:.3f} aspect={aspect:.3f} score={score:.3f}")
        candidates.append((score, int(x), int(y), a, b, angle, rad))

    centers = select_face_centers(candidates, max_faces=max_faces)

    faces: list[Face] = []
    for score, x, y, a, b, angle, rad in centers:
        r_mean, g_mean, b_mean = sample_center_color(img_rgb, x, y)

        faces.append(Face(
            center=(x, y, rad),
            color=(r_mean, g_mean, b_mean),
            sticker=[],
            label=classify((r_mean, g_mean, b_mean), model=get_model())
        ))

    return faces


def sample_center_color(img, x, y, radius=CENTER_RADIUS):
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch_bgr = img[y0:y1, x0:x1]
    b_mean, g_mean, r_mean = patch_bgr.reshape(-1, 3).mean(axis=0)

    return r_mean, g_mean, b_mean


def find_sticker_for_face(face: Face, contours: Any, img: cv2.typing.MatLike) -> None:
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 50000 or area < 500:
            continue

        m = cv2.moments(contour)
        if m['m00'] != 0:
            sticker_x = int(m['m10'] / m['m00'])
            sticker_y = int(m['m01'] / m['m00'])

            face_x = face.center[0]
            face_y = face.center[1]

            distance = math.sqrt((sticker_x - face_x) ** 2 + (sticker_y - face_y) ** 2)

            if face.center[2] * 3.5 > distance > face.center[2]:
                r_mean, g_mean, b_mean = sample_center_color(img, sticker_x, sticker_y)

                sticker = Sticker(
                    center=(sticker_x, sticker_y),
                    color=(r_mean, g_mean, b_mean),
                    label=classify((r_mean, g_mean, b_mean), model=get_model()),
                    angle=None,
                )

                face.sticker.append(sticker)


def show_contours(title:str, contours: Any, img: cv2.typing.MatLike):
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    debug_show(title, img)


def find_contours(path: Path) -> tuple[cv2.typing.MatLike, tuple[int, int], Any]:
    img0 = cv2.imread(str(path))
    assert img0 is not None, f"Image not found: {path}"

    h0, w0 = img0.shape[:2]
    scale = 2000.0 / max(w0, h0)
    img_bgr = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)),
                         cv2.INTER_AREA) if scale < 1 else img0.copy()  # ty:ignore[no-matching-overload]
    debug_show("find_contours: scaled input image", img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_v = hsv[:, :, 2]
    debug_show("find_contours: brightness image", img_v)

    edge_src = cv2.GaussianBlur(img_v, (7, 7), 0)

    sobel_x = cv2.Sobel(edge_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edge_src, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    _, edges = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ty:ignore[no-matching-overload]
    debug_show("find_contours: sobel image", edges)

    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small = [c for c in all_contours if cv2.contourArea(c) < 100]
    if small:
        cv2.drawContours(edges, small, -1, 0, thickness=cv2.FILLED)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
    debug_show("find_contours: cleaned edges image", edges)

    h, w = edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(edges, mask, (0, 0), 255)
    filled = cv2.bitwise_not(edges)
    debug_show("find_contours: floodFill image", filled)

    thr, mask_clean = cv2.threshold(img_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    debug_show("find_contours: threshold image", mask_clean)

    mask_clean = cv2.bitwise_and(mask_clean, filled)
    debug_show("find_contours: threshold image cleaned", mask_clean)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    show_contours("find_contours: found contours", contours, img_bgr.copy())

    if not contours:
        raise ValueError(f"{path.name}: No contours found—check lighting or thresholds.")

    h, w = img_v.shape

    return img_bgr, (h, w), contours


def normalize_radians_np(rad):
    return np.arctan2(np.sin(rad), np.cos(rad))

def find_index_of_stickers(faces: list[Face], img) -> None:

    for face in faces:
        face_angle: list[Entry] = []
        center_x, center_y, _ = face.center

        for other_face in faces:
            if not face is other_face:
                other_center_x, other_center_y, _ = other_face.center

                dx = other_center_x - center_x
                dy = other_center_y - center_y

                angle = math.atan2(dy, dx)

                face_angle.append(Entry(
                    label=other_face.label,
                    angle=angle,
                    sticker=[],
                ))

        stickers = face.sticker
        if not stickers:
            continue

        for sticker in stickers:
            sticker_x, sticker_y = sticker.center
            dx = sticker_x - center_x
            dy = sticker_y - center_y

            angle = math.atan2(dy, dx)
            sticker.angle = angle

            for entry in face_angle[0:1]:
                angle_diff = (angle - entry.angle) % (2 * math.pi)
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

                if angle_diff < MAX_ANGLE_DIFF:

                    diff_angle = normalize_radians_np(angle - entry.angle)

                    new_sticker = Sticker(
                        center=sticker.center,
                        color=sticker.color,
                        label=sticker.label,
                        angle=sticker.angle,
                        dif_angle=diff_angle,
                    )

                    entry.sticker.append(new_sticker)

        if not face_angle or not face_angle[0].sticker:
            # Handle edge case where no stickers were associated
            continue

        face_angle[0].sticker.sort(key=lambda s: s.dif_angle)
        best = face_angle[0].sticker[-1]

        offset_angle = float(best.angle)

        for sticker in stickers:
            sticker.angle = (sticker.angle - offset_angle) % (2 * math.pi)  # ty:ignore[unsupported-operator]

        stickers.sort(key=lambda s: s.angle)

        offset = -2 - (FACE_NEIGHBORS[face.label].index(face_angle[0].label) * 2)

        stickers_ordered = []
        for i in range(len(stickers)):
            idx = (i + offset) % len(stickers)
            stickers_ordered.append(stickers[idx])

        face.sticker = stickers_ordered

    return None


def process_image(path: Path) -> list[Face]:
    img, (h, w), contours = find_contours(path)

    min_radius, max_radius = int(0.02 * min(h, w)), int(0.5 * min(h, w))

    faces = find_face_centers(contours, min_radius, max_radius, max_faces=MAX_FACES, img_rgb=img)

    for face in faces:
        find_sticker_for_face(face, contours, img)
    show_found_stickers(faces, img.copy())

    find_index_of_stickers(faces, img.copy())
    show_sticker_indices(faces, img.copy())

    return faces


def show_sticker_indices(faces: list[Face], img: cv2.typing.MatLike) -> None:
    for face in faces:
        for i, sticker in enumerate(face.sticker):
            cv2.putText(img, f"{i}", sticker.center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    debug_show(f"found sticker index", img)


def show_found_stickers(faces: list[Face], img: cv2.typing.MatLike) -> None:
    for face in faces:
        for sticker in face.sticker:
            cv2.circle(img=img, center=sticker.center, radius=5, color=(0, 255, 0), thickness=-1, lineType=8, shift=0)

    debug_show(f"found stickers", img)


def vote_faces(faces: list[Face]) -> dict[str, list[str]]:
    votes: dict[str, list[dict[str, int]]] = {}
    for face in faces:
        if votes.get(face.label) is None:
            votes[face.label] = []
            for i in range(10):
                votes[face.label].append({})
        for i, sticker in enumerate(face.sticker):
            if sticker.label in votes[face.label][i].keys():
                votes[face.label][i][sticker.label] += 1
            else:
                votes[face.label][i][sticker.label] = 1

    result: dict[str, list[str]] = {}
    for label, vote in votes.items():
        result[label] = []
        for sticker in vote:
            result[label].append(sorted(sticker.items(), key=lambda x: x[1], reverse=True)[0][0])

    return result


def show_result(voted_faces: dict[str, list[str]]) -> None:
    visualize(voted_faces)


IMAGE_DIR = Path("../data/test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    image_paths = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()])
    if not image_paths:
        raise SystemExit(f"No image files found in {IMAGE_DIR}")

    faces: list[Face] = []
    for path in image_paths:
        faces += process_image(path)

    voted_faces: dict[str, list[str]] = vote_faces(faces)
    show_result(voted_faces)

if __name__ == "__main__":
    main()
