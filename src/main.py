import cv2
import math
import numpy as np

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


def make_kernel(k):
    k = max(1, int(k) | 1)  # odd >=1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

path = "../data/faces/blue.jpeg"  # <-- set your file
img0 = cv2.imread(path)
assert img0 is not None, "Image not found"

# normalize size
h0, w0 = img0.shape[:2]
scale = 1400.0 / max(w0, h0)
img = cv2.resize(img0, (int(w0*scale), int(h0*scale)), cv2.INTER_AREA) if scale < 1 else img0.copy()
show("00 - input", img)

# non-black mask via Otsu on V channel
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

V = hsv[:, :, 2]
thr, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
new_thr, non_black = cv2.threshold(V, thr-50, 255, cv2.THRESH_BINARY)  # add bias

show(f"02 - non-black mask (Otsu thr={new_thr:.1f})", non_black)

mask_clean = cv2.morphologyEx(non_black, cv2.MORPH_OPEN, make_kernel(5), iterations=5)
show("03 - cleaned non-black (no bridging)", mask_clean)

cnts, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
assert cnts, "No contours found—check lighting or thresholds."

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

assert best is not None, "Center circle not found—adjust size range."
cx, cy, R = best


img_test = img.copy()

selected_contours = []

for i, contour in enumerate(cnts):

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
            print(f"Contour {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

            cv2.drawContours(img_test, [contour], 0, (0, 0, 255), 5)
            cv2.putText(img_test, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


all_pts = np.vstack([c.reshape(-1, 2) for c in selected_contours]).astype(np.int32)

hull = cv2.convexHull(all_pts)
cv2.polylines(img_test, [hull], isClosed=True, color=(0, 255, 0), thickness=3)

show("03b - contours with shape labels", img_test)
