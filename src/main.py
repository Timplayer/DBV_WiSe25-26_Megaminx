import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.cluster import KMeans

SHOW_INTERMEDIATE = False  # set True to view OpenCV debug windows

def show(title, img, max_w=1200, max_h=800):
    """Show an image (auto-resized) and wait for a key.
       Press any key to continue; ESC exits."""

    if not SHOW_INTERMEDIATE:
        return

    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        raise SystemExit


def make_kernel(k):
    k = max(1, int(k) | 1)  # odd >=1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def plot_contour_colors(contour_colors):
    """Visualize detected contour colors across all images as horizontal swatches."""
    if not contour_colors:
        print("No contours met selection criteria; skipping color plot.")
        return

    labels = [
        f"{c['file']} #{c['index']}: H={c['hsv'][0]:.0f} S={c['hsv'][1]:.0f} V={c['hsv'][2]:.0f}"
        for c in contour_colors
    ]
    colors = [c["rgb"] for c in contour_colors]
    positions = np.arange(len(contour_colors))

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(contour_colors))))
    ax.barh(positions, np.ones(len(contour_colors)), color=colors, edgecolor="black")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Detected contour colors across images")
    ax.invert_yaxis()  # keep first contour at top
    fig.tight_layout()
    plt.show()

def plot_hsv_2d(contour_colors):
    """Plot a 2D Hue-Saturation scatter colored by the original RGB."""
    if not contour_colors:
        print("No contour colors available for 2D plot.")
        return
    H = [c["hsv"][0] for c in contour_colors]
    S = [c["hsv"][1] for c in contour_colors]
    colors = [c["rgb"] for c in contour_colors]
    labels = [f"{c['file']}#{c['index']}" for c in contour_colors]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(H, S, c=colors, edgecolors="black", alpha=0.8)
    for h, s, label in zip(H, S, labels):
        ax.text(h, s, label, fontsize=8, ha="center", va="center")
    ax.set_xlabel("Hue (0-179)")
    ax.set_ylabel("Saturation (0-255)")
    ax.set_title("Hue vs Saturation for detected contours")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()

def plot_hsv_3d(contour_colors):
    """Plot a 3D Hue-Saturation-Value scatter colored by the original RGB."""
    if not contour_colors:
        print("No contour colors available for 3D plot.")
        return
    H = [c["hsv"][0] for c in contour_colors]
    S = [c["hsv"][1] for c in contour_colors]
    V = [c["hsv"][2] for c in contour_colors]
    colors = [c["rgb"] for c in contour_colors]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(H, S, V, c=colors, edgecolors="black", depthshade=True)
    ax.set_xlabel("Hue (0-179)")
    ax.set_ylabel("Saturation (0-255)")
    ax.set_zlabel("Value (0-255)")
    ax.set_title("HSV scatter for detected contours")
    plt.tight_layout()
    plt.show()

def cluster_hsv(contour_colors, n_clusters=12):
    """Cluster HSV values; enforce max 11 samples per cluster and return filtered results."""
    if not contour_colors:
        print("No contour colors available for clustering.")
        return None
    data = np.array([c["hsv"] for c in contour_colors], dtype=np.float32)
    n_eff = min(n_clusters, len(contour_colors))
    if n_eff < n_clusters:
        print(f"Only {len(contour_colors)} samples; clustering into {n_eff} clusters instead of {n_clusters}.")
    kmeans = KMeans(n_clusters=n_eff, n_init="auto", random_state=0)
    labels_full = kmeans.fit_predict(data)

    cluster_names = {}
    kept_pairs = []  # (global_idx, cluster_id)
    for k in range(n_eff):
        idxs = np.where(labels_full == k)[0]
        files = [contour_colors[idx]["file"] for idx in idxs]
        if files:
            name = Counter(files).most_common(1)[0][0]
        else:
            name = f"Cluster {k}"
        cluster_names[k] = name

        if len(idxs) > 11:
            center = kmeans.cluster_centers_[k]
            distances = np.linalg.norm(data[idxs] - center, axis=1)
            order = np.argsort(distances)
            keep = idxs[order[:11]]
            dropped = len(idxs) - len(keep)
            print(f"Cluster {k} -> {name}: keeping 11 of {len(idxs)} (dropped {dropped})")
        else:
            keep = idxs
            print(f"Cluster {k} -> {name}: {len(idxs)} samples")
            if len(idxs) < 11:
                print(f"  Warning: cluster {k} has {len(idxs)} < 11 samples.")

        kept_pairs.extend([(int(idx), k) for idx in keep])

    # Build filtered outputs capped at 11 per cluster
    filtered_colors = [contour_colors[idx] for idx, _ in kept_pairs]
    labels_filtered = [k for _, k in kept_pairs]

    return filtered_colors, labels_filtered, cluster_names, kmeans.cluster_centers_

def plot_hsv_3d_clustered(contour_colors, labels, cluster_names, centers):
    """Plot a 3D scatter colored by cluster assignment with labeled centroids."""
    if labels is None or not contour_colors:
        print("No clustering results available for 3D clustered plot.")
        return
    H = [c["hsv"][0] for c in contour_colors]
    S = [c["hsv"][1] for c in contour_colors]
    V = [c["hsv"][2] for c in contour_colors]
    labels_arr = np.array(labels)
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(int(lbl) % 20) for lbl in labels_arr]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(H, S, V, c=colors, edgecolors="black", depthshade=True, alpha=0.85)

    # Plot centroids
    for idx, center in enumerate(centers):
        ax.scatter(center[0], center[1], center[2], marker="X", s=120, color=cmap(int(idx) % 20), edgecolors="black")
        name = cluster_names.get(idx, f"Cluster {idx}")
        ax.text(center[0], center[1], center[2], name, fontsize=9, weight="bold")

    legend_labels = [f"{idx} - {cluster_names.get(idx, f'Cluster {idx}')}" for idx in sorted(cluster_names.keys())]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(int(idx) % 20),
                          markeredgecolor="black", markersize=8, label=legend_labels[idx])
               for idx in sorted(cluster_names.keys())]
    ax.legend(handles=handles, title="Cluster -> filename", loc="best", fontsize=8)
    ax.set_xlabel("Hue (0-179)")
    ax.set_ylabel("Saturation (0-255)")
    ax.set_zlabel("Value (0-255)")
    ax.set_title("HSV clusters (3D)")
    plt.tight_layout()
    plt.show()

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
    thr, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    new_thr, non_black = cv2.threshold(V, thr-50, 255, cv2.THRESH_BINARY)  # add bias

    show(f"02 - non-black mask ({path.name}) (Otsu thr={new_thr:.1f})", non_black)

    mask_clean = cv2.morphologyEx(non_black, cv2.MORPH_OPEN, make_kernel(5), iterations=5)
    show(f"03 - cleaned non-black ({path.name})", mask_clean)

    cnts, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        print(f"{path.name}: No contours found—check lighting or thresholds.")
        return []

    H, W = V.shape
    minR, maxR = int(0.02*min(H,W)), int(0.15*min(H,W))  # expected center range

    best = None
    best_score = -1

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < minR or r > maxR:
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circularity = 4 * np.pi * area / (per * per)

        circle_fill = min(area / (np.pi*r*r), (np.pi*r*r) / max(area,1))
        score = 0.7*circularity + 0.3*circle_fill
        if score > best_score:
            best_score, best = (score, (int(x), int(y), int(r)))

    if best is None:
        print(f"{path.name}: Center circle not found—adjust size range.")
        return []

    cx, cy, R = best

    img_test = img.copy()

    selected_contours = []
    contour_colors = []

    for i, contour in enumerate(cnts):
        if cv2.contourArea(contour) > 50000:
            continue
        if cv2.contourArea(contour) < 500:
            continue

        M = cv2.moments(contour)
        if M['m00'] != 0:
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])

            d_x = x - cx
            d_y = y - cy

            distance = math.sqrt(d_x**2 + d_y**2)

            if distance < R * 3:
                selected_contours.append(contour)

                h, w = hsv.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, contour, contourIdx=-1, color=255, thickness=cv2.FILLED)

                h_mean, s_mean, v_mean, _ = cv2.mean(hsv, mask=mask)
                b_mean, g_mean, r_mean, _ = cv2.mean(img, mask=mask)
                print(f"{path.name} - Contour {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
                contour_colors.append({
                    "file": path.name,
                    "index": i,
                    "hsv": (h_mean, s_mean, v_mean),
                    "rgb": (r_mean / 255.0, g_mean / 255.0, b_mean / 255.0),
                })

                cv2.drawContours(img_test, [contour], 0, (0, 0, 255), 5)
                cv2.putText(img_test, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if selected_contours:
        all_pts = np.vstack([c.reshape(-1, 2) for c in selected_contours]).astype(np.int32)
        hull = cv2.convexHull(all_pts)
        cv2.polylines(img_test, [hull], isClosed=True, color=(0, 255, 0), thickness=3)

    show(f"03b - contours with shape labels ({path.name})", img_test)
    return contour_colors

IMAGE_DIR = Path("../data/faces")
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

    cv2.destroyAllWindows()
    plot_contour_colors(all_contour_colors)
    plot_hsv_2d(all_contour_colors)
    plot_hsv_3d(all_contour_colors)
    clustering = cluster_hsv(all_contour_colors, n_clusters=12)
    if clustering is not None:
        filtered_colors, labels, cluster_names, centers = clustering
        plot_hsv_3d_clustered(filtered_colors, labels, cluster_names, centers)

if __name__ == "__main__":
    main()
