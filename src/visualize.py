import cv2
import numpy as np
import math

from src.canonical import FACE_NEIGHBORS, LABEL_TO_COLOR

ANGLE = math.radians(72)   # unfold angle
SIDE = 500                  # edge length in pixels
HEIGHT, WIDTH = 8000, 8000


def pentagon_inner_ring(pentagon, radius_ratio=0.6):
    """
    Place 10 points evenly spaced in a ring inside a pentagon.

    pentagon     : (5,2) ndarray of vertices (clockwise or CCW)
    radius_ratio : distance from centroid relative to max centroid-vertex distance (0<r<1)

    returns : (10,2) ndarray of points
    """
    pentagon = np.asarray(pentagon, dtype=float)
    if pentagon.shape != (5,2):
        raise ValueError("pentagon must be (5,2) array")

    # centroid
    centroid = pentagon.mean(axis=0)

    dy = centroid[1] - pentagon[0][1]
    dx = centroid[0] - pentagon[0][0]
    angle_offset = math.atan2(dy, dx)

    # maximal radius from centroid to a vertex
    max_r = np.max(np.linalg.norm(pentagon - centroid, axis=1))

    # actual radius for the ring
    r = max_r * radius_ratio

    points = []
    for i in range(10):
        angle = angle_offset + (2 * math.pi * i / 10)  # 10 points equally spaced
        x = centroid[0] + r * math.cos(angle)
        y = centroid[1] + r * math.sin(angle)
        points.append([x, y])

    return np.array(points).astype(int)


def regular_pentagon_from_edge(p0, p1, outward=True):
    """
    Construct a regular pentagon where p0 -> p1 is an exact edge.
    Vertex order is counterclockwise if outward=True.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    L = np.linalg.norm(p1 - p0)
    if L == 0:
        raise ValueError("p0 and p1 must be distinct")

    # Direction of first edge
    e = (p1 - p0) / L

    # Exterior angle of regular pentagon
    exterior = 2 * math.pi / 5  # 72°

    if not outward:
        exterior = -exterior

    # Circumradius
    R = L / (2 * math.sin(math.pi / 5))

    # Compute center of circumcircle
    mid = (p0 + p1) / 2
    perp = np.array([-e[1], e[0]])

    h = math.sqrt(R**2 - (L / 2)**2)
    center = mid + perp * h if outward else mid - perp * h

    # Generate vertices
    angle0 = math.atan2(p0[1] - center[1], p0[0] - center[0])

    vertices = []
    for i in range(5):
        angle = angle0 + i * exterior
        v = center + R * np.array([math.cos(angle), math.sin(angle)])
        vertices.append(v)

    return np.array(vertices), center


def visualize(faces: dict[str, list[str]]):
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    pentagon = [(6000, 700), (6000, 700 - SIDE)]
    prev_label = ""
    preprev_label = ""
    for j, (face, neighbor) in enumerate(FACE_NEIGHBORS.items()):

        i = 0
        if prev_label in FACE_NEIGHBORS.keys() and face in FACE_NEIGHBORS[prev_label] and preprev_label in FACE_NEIGHBORS[prev_label]:
            i = FACE_NEIGHBORS[prev_label].index(face)-FACE_NEIGHBORS[prev_label].index(preprev_label)
        elif j > 1:
            break

        color_offset = -5
        if prev_label in FACE_NEIGHBORS[face]:
            color_offset = -5 + FACE_NEIGHBORS[face].index(prev_label) * 2
        p0 = pentagon[(i + 1) % len(pentagon)]
        p1 = pentagon[(i) % len(pentagon)]
        preprev_label = prev_label
        prev_label = face

        pentagon, center = regular_pentagon_from_edge(p0, p1)
        pts = pentagon.astype(int)
        cv2.polylines(img, [pts], True, (255, 255, 255), int(SIDE/50)),
        cv2.circle(img, center.astype(int), 70, LABEL_TO_COLOR[face], -1)
        for i, point in enumerate(pentagon_inner_ring(pentagon)):
            #cv2.putText(img, str((i+color_offset)%10), point, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            if face in faces.keys():
                cv2.circle(img, point, 50, LABEL_TO_COLOR[faces[face][(i+color_offset)%10]], -1)

    cv2.namedWindow("Dodecahedron Net (No Overlaps)", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Dodecahedron Net (No Overlaps)", img)
    cv2.resizeWindow("Dodecahedron Net (No Overlaps)", 500, 500)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
