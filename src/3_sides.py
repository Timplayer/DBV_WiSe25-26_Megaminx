import cv2
import math
import numpy as np
from pathlib import Path

MAX_FACES = 3
ASPECT_MIN = 0.5  # how round an ellipse must be (1.0 == perfect circle)
# BGR colors used to visualize each detected face separately
FACE_COLORS = [
    (0, 0, 255),
    (0, 200, 0),
    (255, 100, 0),
    (0, 200, 200),
]

def make_kernel(k):
    k = max(1, int(k) | 1)  # odd >=1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

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

def find_face_centers(contours, minR, maxR, max_faces=MAX_FACES):
    """Return up to `max_faces` center candidates; allow ellipses (flattened circles)."""
    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 4000:
            continue

        if len(c) < 5:
            continue  # need enough points to fit an ellipse

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)
        a, b = 0.5 * MA, 0.5 * ma  # semi-axes for cv2.ellipse
        rad = 0.5 * (a + b)        # average semi-axis for range/NMS
        if rad < minR or rad > maxR:
            continue

        ellipse_area = math.pi * a * b
        fill = min(area / ellipse_area, ellipse_area / max(area, 1))
        aspect = min(a, b) / max(a, b)

        score = fill
        print(f"Candidate {i}: center=({x:.1f},{y:.1f}) axes=({a:.1f},{b:.1f}) angle={angle:.1f} area={area:.1f} fill={fill:.3f} aspect={aspect:.3f} score={score:.3f}")
        candidates.append((score, int(x), int(y), a, b, angle, rad))


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

def process_image(path):
    """Run contour detection for a single image path and return contour color info."""
    img0 = cv2.imread(str(path))
    assert img0 is not None, f"Image not found: {path}"

    # normalize size
    h0, w0 = img0.shape[:2]
    scale = 1400.0 / max(w0, h0)
    img = cv2.resize(img0, (int(w0*scale), int(h0*scale)), cv2.INTER_AREA) if scale < 1 else img0.copy()
    show(f"00 - input ({path.name})", img)

    # non-black mask via Otsu on V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    V = hsv[:, :, 2]
    thr, non_black = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #new_thr, non_black = cv2.threshold(V, thr, 255, cv2.THRESH_BINARY)  # add bias

    show(f"02 - non-black mask ({path.name}) (Otsu thr={thr:.1f})", non_black)

    mask_clean = cv2.morphologyEx(non_black, cv2.MORPH_OPEN, make_kernel(3), iterations=5)
    show(f"03 - cleaned non-black ({path.name})", mask_clean)

    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        print(f"{path.name}: No contours found—check lighting or thresholds.")
        return []

    H, W = V.shape
    minR, maxR = int(0.02*min(H,W)), int(0.5*min(H,W))  # expected center range

    centers = find_face_centers(cnts, minR, maxR, MAX_FACES)
    if not centers:
        print(f"{path.name}: No face centers found—adjust size range.")
        return []

    if len(centers) < MAX_FACES:
        print(f"{path.name}: Only found {len(centers)} face center(s); proceeding with those.")

    faces = []
    for score, cx, cy, a, b, angle, rad in centers:
        faces.append({
            "center": (cx, cy),
            "axes": (a, b),
            "angle": angle,
            "rad": rad,
            "score": score,
            "contours": []
        })

    img_test = img.copy()

    contour_colors = []

    for i, contour in enumerate(cnts):
        if cv2.contourArea(contour) > 50000:
            continue
        if cv2.contourArea(contour) < 1000:
            continue

        M = cv2.moments(contour)
        if M['m00'] != 0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

            # assign contour to the nearest face center
            distances = [math.sqrt((x - fx)**2 + (y - fy)**2) for fx, fy in [f["center"] for f in faces]]
            face_idx = int(np.argmin(distances))
            distance = distances[face_idx]
            face = faces[face_idx]

            if distance < face["rad"] * 4:
                face["contours"].append(contour)

                h, w = hsv.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, contour, contourIdx=-1, color=255, thickness=cv2.FILLED)

                h_mean, s_mean, v_mean, _ = cv2.mean(hsv, mask=mask)
                b_mean, g_mean, r_mean, _ = cv2.mean(img, mask=mask)
                print(f"{path.name} - Contour {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
                contour_colors.append({
                    "file": path.name,
                    "index": i,
                    "face": face_idx,
                    "hsv": (h_mean, s_mean, v_mean),
                    "rgb": (r_mean / 255.0, g_mean / 255.0, b_mean / 255.0),
                })

                color = FACE_COLORS[face_idx % len(FACE_COLORS)]
                cv2.drawContours(img_test, [contour], 0, color, 3)
                cv2.putText(img_test, f"{face_idx}:{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    for idx, face in enumerate(faces):
        color = FACE_COLORS[idx % len(FACE_COLORS)]
        cx, cy = face["center"]
        ax, by = face["axes"]
        cv2.ellipse(img_test, (cx, cy), (int(ax), int(by)), face["angle"], 0, 360, color, 2)
        cv2.putText(img_test, f"Face {idx}", (cx - 20, cy - int(face["rad"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if face["contours"]:
            all_pts = np.vstack([c.reshape(-1, 2) for c in face["contours"]]).astype(np.int32)
            hull = cv2.convexHull(all_pts)
            cv2.polylines(img_test, [hull], isClosed=True, color=color, thickness=3)

    show(f"03b - contours with shape labels ({path.name})", img_test)
    return contour_colors

IMAGE_DIR = Path("../data/3_faces")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    image_paths = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
    if not image_paths:
        raise SystemExit(f"No image files found in {IMAGE_DIR}")

    all_contour_colors = []
    for path in image_paths:
        print(f"\nProcessing {path.name}...")
        contour_colors = process_image(path)
        all_contour_colors.extend(contour_colors)


if __name__ == "__main__":
    main()
