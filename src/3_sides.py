import cv2
import math
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

from canonical import FACE_ORDER, RING_DIRECTION, FACE_NEIGHBORS
from color_mapping import get_model, classify, load_color_samples

MAX_FACES = 3
ASPECT_MIN = 0.5  # how round an ellipse must be (1.0 == perfect circle)
CENTER_RADIUS = 6  # sample radius (pixels) around contour center
STICKERS_PER_FACE = 11
RING_SIZE = STICKERS_PER_FACE - 1
DEBUG_NEIGHBOR_ALIGN = True

# BGR colors used to visualize each detected face separately
FACE_COLORS = [
    (0, 0, 255),
    (0, 200, 0),
    (255, 100, 0),
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

def sample_center_color(img, x, y, radius=CENTER_RADIUS):
    h, w = img.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch_bgr = img[y0:y1, x0:x1]
    b_mean, g_mean, r_mean = patch_bgr.reshape(-1, 3).mean(axis=0)

    return r_mean, g_mean, b_mean

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

def compute_center_guess(stickers):
    if not stickers:
        return None
    weights = np.array([max(float(s.get("area", 0.0)), 1.0) for s in stickers], dtype=np.float32)
    coords = np.array([s["center"] for s in stickers], dtype=np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        mean = coords.mean(axis=0)
        return float(mean[0]), float(mean[1])
    weighted = (coords.T * weights).T
    mean = weighted.sum(axis=0) / weight_sum
    return float(mean[0]), float(mean[1])

def angle_diff(a, b):
    diff = abs(a - b)
    return min(diff, 2 * math.pi - diff)

def angle_to_point(cx, cy, px, py):
    angle = math.atan2(py - cy, px - cx)
    if angle < 0:
        angle += 2 * math.pi
    return angle

def order_face_stickers(face, path_name):
    stickers = face["stickers"]
    if not stickers:
        return []

    stickers_sorted = sorted(stickers, key=lambda s: s["distance"])
    kept = stickers_sorted[:STICKERS_PER_FACE]

    if len(stickers) > STICKERS_PER_FACE:
        print(f"{path_name} - Face {face['index']}: pruning {len(stickers)} -> {len(kept)}")
    elif len(stickers) < STICKERS_PER_FACE:
        print(f"{path_name} - Face {face['index']}: only {len(stickers)} stickers (expected {STICKERS_PER_FACE})")

    kept_ids = {id(s) for s in kept}
    for s in stickers:
        s["kept"] = id(s) in kept_ids

    if not kept:
        return []

    center_guess = compute_center_guess(kept)
    if center_guess is None:
        return []
    face["center_refined"] = center_guess
    cx, cy = center_guess

    for s in kept:
        dx = s["center"][0] - cx
        dy = s["center"][1] - cy
        s["distance_refined"] = math.hypot(dx, dy)

    kept_by_dist = sorted(kept, key=lambda s: s["distance_refined"])
    center = kept_by_dist[0]
    if len(kept_by_dist) > 1:
        d0 = kept_by_dist[0]["distance_refined"]
        d1 = kept_by_dist[1]["distance_refined"]
        distances = [s["distance_refined"] for s in kept_by_dist]
        median_dist = float(np.median(distances))
        ambiguous = (d1 - d0) < max(2.0, 0.1 * face["rad"]) or (median_dist > 0 and d0 > 0.5 * median_dist)
        if ambiguous:
            candidates = kept_by_dist[: min(3, len(kept_by_dist))]
            center = max(candidates, key=lambda s: s.get("area", 0.0))
            print(
                f"{path_name} - Face {face['index']}: center ambiguous "
                f"(d0={d0:.1f}, d1={d1:.1f}, median={median_dist:.1f}), "
                f"selected by area."
            )

    ring = []
    for s in kept:
        if s is center:
            continue
        dx = s["center"][0] - cx
        dy = s["center"][1] - cy
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        s["angle"] = angle
        ring.append(s)

    ring_sorted = sorted(ring, key=lambda s: s["angle"])
    ordered = [center] + ring_sorted

    for idx, s in enumerate(ordered):
        s["order"] = idx
        s["is_center"] = (idx == 0)
    center["angle"] = 0.0

    face["ordered"] = ordered
    face["label"] = center.get("label")
    return ordered

def annotate_face_neighbors(faces):
    for face in faces:
        ordered = face.get("ordered", [])
        if len(ordered) < 2:
            face["neighbor_angles"] = {}
            face["neighbor_edges"] = {}
            face["neighbor_angles_label"] = {}
            face["neighbor_edges_label"] = {}
            continue
        ring = ordered[1:]
        cx, cy = face.get("center_refined", face["center"])
        neighbor_angles = {}
        neighbor_edges = {}
        neighbor_angles_label = {}
        neighbor_edges_label = {}
        for other in faces:
            if other is face:
                continue
            ox, oy = other.get("center_refined", other["center"])
            direction = angle_to_point(cx, cy, ox, oy)
            neighbor_angles[other["index"]] = direction
            other_label = other.get("label")
            if other_label:
                neighbor_angles_label[other_label] = direction
            best_idx = None
            best_diff = None
            for idx, sticker in enumerate(ring):
                angle = sticker.get("angle")
                if angle is None:
                    continue
                diff = angle_diff(direction, angle)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            if best_idx is not None:
                neighbor_edges[other["index"]] = best_idx
                if other_label:
                    neighbor_edges_label[other_label] = best_idx
        face["neighbor_angles"] = neighbor_angles
        face["neighbor_edges"] = neighbor_edges
        face["neighbor_angles_label"] = neighbor_angles_label
        face["neighbor_edges_label"] = neighbor_edges_label

def most_common(values):
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts.items(), key=lambda item: item[1])[0]

def derive_canonical_rotations(face_map):
    canonical = {}
    for label, face in face_map.items():
        neighbors = FACE_NEIGHBORS.get(label)
        if not neighbors:
            continue
        observed = face.get("neighbor_edges_label", {})
        if not observed:
            continue
        offsets = []
        if DEBUG_NEIGHBOR_ALIGN:
            print(f"\nCanonical align for {label}:")
            print(f"  Observed edges: {observed}")
            print(f"  Expected neighbor order: {neighbors}")
        for idx, neighbor_label in enumerate(neighbors):
            if neighbor_label not in observed:
                continue
            expected = (idx * 2) % RING_SIZE
            offsets.append((observed[neighbor_label] - expected) % RING_SIZE)
            if DEBUG_NEIGHBOR_ALIGN:
                print(
                    f"  Neighbor {neighbor_label}: observed={observed[neighbor_label]} "
                    f"expected={expected} offset={(observed[neighbor_label] - expected) % RING_SIZE}"
                )
        if offsets:
            canonical[label] = most_common(offsets)
            if DEBUG_NEIGHBOR_ALIGN:
                print(f"  -> chosen canonical rotation offset: {canonical[label]}")
    return canonical

def build_components(constraints, labels):
    graph = defaultdict(set)
    for a_label, _, b_label, _, _ in constraints:
        graph[a_label].add(b_label)
        graph[b_label].add(a_label)

    components = []
    seen = set()
    for label in labels:
        if label in seen:
            continue
        queue = deque([label])
        seen.add(label)
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in graph.get(cur, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        components.append(comp)
    return components

def apply_canonical_offsets(rotations, canonical_rotations, face_map, constraints):
    labels = set(face_map.keys()) | set(rotations.keys()) | set(canonical_rotations.keys())
    if not labels:
        return rotations
    components = build_components(constraints, labels)
    updated = dict(rotations)
    for comp in components:
        offsets = []
        for label in comp:
            if label in canonical_rotations and label in updated:
                offsets.append((canonical_rotations[label] - updated[label]) % RING_SIZE)
        offset = most_common(offsets) if offsets else 0
        if DEBUG_NEIGHBOR_ALIGN:
            print(f"\nComponent {comp}:")
            print(f"  offsets={offsets} -> applied offset={offset}")
        for label in comp:
            if label in updated:
                updated[label] = (updated[label] + offset) % RING_SIZE
            elif label in canonical_rotations:
                updated[label] = canonical_rotations[label]
    return updated

def build_constraints(faces, image_name):
    constraints = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            fa = faces[i]
            fb = faces[j]
            if not fa.get("label") or not fb.get("label"):
                continue
            ia = fa.get("neighbor_edges", {}).get(fb["index"])
            ib = fb.get("neighbor_edges", {}).get(fa["index"])
            if ia is None or ib is None:
                continue
            constraints.append((fa["label"], ia, fb["label"], ib, image_name))
    return constraints

def solve_rotations(constraints):
    adj = defaultdict(list)
    for a_label, ia, b_label, ib, src in constraints:
        adj[a_label].append((b_label, ia, ib, src))
        adj[b_label].append((a_label, ib, ia, src))

    rotations = {}
    for start in adj.keys():
        if start in rotations:
            continue
        rotations[start] = 0
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for other, i_cur, i_other, src in adj[cur]:
                rot_other = (rotations[cur] + i_cur - i_other) % RING_SIZE
                if other not in rotations:
                    rotations[other] = rot_other
                    queue.append(other)
                elif rotations[other] != rot_other:
                    print(
                        f"Rotation conflict {cur} -> {other} from {src}: "
                        f"{rotations[other]} vs {rot_other}"
                    )
    return rotations

def pick_best_faces(face_obs):
    best = {}
    for face in face_obs:
        label = face.get("label")
        if not label:
            continue
        ordered = face.get("ordered", [])
        score = len(ordered)
        prev = best.get(label)
        if prev is None or score > prev["score"]:
            best[label] = {"face": face, "score": score}
    return {label: entry["face"] for label, entry in best.items()}

def canonical_ring_labels(ordered, rotation):
    if not ordered:
        return ["?"] * STICKERS_PER_FACE

    center = ordered[0].get("label") or "?"
    ring = [s.get("label") or "?" for s in ordered[1:]]
    if len(ring) < RING_SIZE:
        ring += ["?"] * (RING_SIZE - len(ring))
    elif len(ring) > RING_SIZE:
        ring = ring[:RING_SIZE]

    if RING_DIRECTION == "counterclockwise":
        ring = list(reversed(ring))

    rot = rotation % RING_SIZE
    rotated = [ring[(idx - rot) % RING_SIZE] for idx in range(RING_SIZE)]
    return [center] + rotated

def resolve_face_order(face_map):
    if FACE_ORDER:
        return list(FACE_ORDER)
    try:
        _, _, labels = load_color_samples()
        if labels:
            return list(labels)
    except Exception:
        pass
    if face_map:
        return sorted(face_map.keys())
    return []

def format_state_string(face_map, face_order, rotations):
    parts = []
    for label in face_order:
        face = face_map.get(label)
        if face:
            rot = rotations.get(label, 0)
            labels = canonical_ring_labels(face.get("ordered", []), rot)
        else:
            labels = ["?"] * STICKERS_PER_FACE
        parts.append(f"{label}:{','.join(labels)}")
    return " | ".join(parts)


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

def process_image(path):
    """Run contour detection for a single image path and return faces and contour colors."""
    img0 = cv2.imread(str(path))

    assert img0 is not None, f"Image not found: {path}"

    model = get_model()

    h0, w0 = img0.shape[:2]
    scale = 2000.0 / max(w0, h0)
    img = cv2.resize(img0, (int(w0 * scale), int(h0 * scale)), cv2.INTER_AREA) if scale < 1 else img0.copy()

    show(f"00 - input ({path.name})", img)

    # non-black mask via Otsu on V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    V = hsv[:, :, 2]

    #show(f"02 - non-black mask ({path.name}) (Otsu thr={thr:.1f})", non_black)

    edge_src = cv2.GaussianBlur(V, (7, 7), 0)

    show(f"01 - blurred V channel ({path.name})", edge_src)

    sobel_x = cv2.Sobel(edge_src, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(edge_src, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))

    show(f"01 - sobel magnitude ({path.name})", sobel_mag)

    _, edges = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show(f"01 - raw edges ({path.name})", edges)

    all_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    small = [c for c in all_contours if cv2.contourArea(c) < 100]
    if small:
        cv2.drawContours(edges, small, -1, 0, thickness=cv2.FILLED)

    show(f"01 - raw edges ({path.name})", edges)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)

    show(f"02 - edges ({path.name})", edges)

    h, w = edges.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(edges, mask, (0, 0), 255)
    filled = cv2.bitwise_not(edges)

    thr, mask_clean = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show(f"02 - V channel thresholded at {thr:.1f} ({path.name})", mask_clean)

    mask_clean = cv2.bitwise_and(mask_clean, filled)

    show(f"03a - edges ({path.name})", filled)
    show(f"02 - non-black mask ({path.name})", mask_clean)

    center_contours, _ = cv2.findContours(filled, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print(f"{path.name}: No contours found—check lighting or thresholds.")
        return [], []

    H, W = V.shape
    minR, maxR = int(0.02 * min(H,W)), int(0.5 * min(H,W))  # expected center range

    centers = find_face_centers(center_contours, minR, maxR, max_faces=MAX_FACES)
    if not centers:
        print(f"{path.name}: No face centers found—adjust size range.")
        return [], []

    if len(centers) < MAX_FACES:
        print(f"{path.name}: Only found {len(centers)} face center(s); proceeding with those.")

    faces = []
    for score, cx, cy, a, b, angle, rad in centers:
        faces.append({
            "index": len(faces),
            "center": (cx, cy),
            "axes": (a, b),
            "angle": angle,
            "rad": rad,
            "score": score,
            "contours": [],
            "stickers": [],
            "ordered": [],
            "label": None,
            "neighbor_angles": {},
            "neighbor_edges": {},
        })

    img_test = img.copy()
    img_fill = img.copy()

    contour_colors = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > 50000 or area < 500:
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

            if distance < face["rad"] * 3.5:
                face["contours"].append(contour)

                r_mean, g_mean, b_mean = sample_center_color(img, x, y)
                rgb = (int(round(r_mean)), int(round(g_mean)), int(round(b_mean)))

                label = classify(rgb, model)

                print(f"{path.name} - Contour {i}: ({rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}) -> {label}")

                entry = {
                    "file": path.name,
                    "index": i,
                    "face": face_idx,
                    "center": (x, y),
                    "area": area,
                    "distance": distance,
                    "rgb": (r_mean / 255.0, g_mean / 255.0, b_mean / 255.0),
                    "label": label,
                    "kept": False,
                    "order": None,
                    "is_center": False,
                    "angle": None,
                }
                contour_colors.append(entry)
                face["stickers"].append(entry)

                color = FACE_COLORS[face_idx % len(FACE_COLORS)]
                cv2.drawContours(img_test, [contour], 0, color, 3)
                cv2.putText(img_test, f"{label}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.drawContours(img_fill, [contour], 0, (rgb[2], rgb[1], rgb[0]), thickness=cv2.FILLED)

    for face in faces:
        order_face_stickers(face, path.name)
    annotate_face_neighbors(faces)

    for idx, face in enumerate(faces):
        color = FACE_COLORS[idx % len(FACE_COLORS)]
        cx, cy = face["center"]
        ax, by = face["axes"]
        for sticker in face.get("ordered", []):
            if sticker.get("order") is None:
                continue
            sx, sy = sticker["center"]
            label = str(sticker["order"])
            cv2.putText(img_test, label, (sx - 8, sy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img_test, label, (sx - 8, sy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.ellipse(img_test, (cx, cy), (int(ax), int(by)), face["angle"], 0, 360, color, 2)
        cv2.putText(img_test, f"Face {idx}", (cx - 20, cy - int(face["rad"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if face["contours"]:
            all_pts = np.vstack([c.reshape(-1, 2) for c in face["contours"]]).astype(np.int32)
            hull = cv2.convexHull(all_pts)
            cv2.polylines(img_test, [hull], isClosed=True, color=color, thickness=3)

    show(f"03b - contours with shape labels ({path.name})", img_test)
    show(f"03c - contours filled with mean color ({path.name})", img_fill)
    return faces, contour_colors

IMAGE_DIR = Path("../data/test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main():
    image_paths = sorted([p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
    if not image_paths:
        raise SystemExit(f"No image files found in {IMAGE_DIR}")

    all_faces = []
    all_contour_colors = []
    constraints = []
    for path in image_paths:
        print(f"\nProcessing {path.name}...")
        faces, contour_colors = process_image(path)
        all_faces.extend(faces)
        all_contour_colors.extend(contour_colors)
        constraints.extend(build_constraints(faces, path.name))

    if not all_faces:
        print("No faces detected across images; skipping final state.")
        return

    rotations = solve_rotations(constraints)
    face_map = pick_best_faces(all_faces)
    canonical_rotations = derive_canonical_rotations(face_map)
    rotations = apply_canonical_offsets(rotations, canonical_rotations, face_map, constraints)
    face_order = resolve_face_order(face_map)

    if not face_order:
        print("No face order available; skipping final state.")
        return
    state_string = format_state_string(face_map, face_order, rotations)
    print("\nFINAL_STATE:")
    print(state_string)


if __name__ == "__main__":
    main()
