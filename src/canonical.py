"""
Canonical face order and (optional) neighbor layout for the final state output.

Fill FACE_ORDER with your 12 face labels (center labels), in the order you want
the final state string to list them. Leave it empty to fall back to the order
from data/colors.csv (its header order).
"""

# Example template (replace with your actual labels)
# FACE_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"]
FACE_ORDER = []

# Optional: ring direction used when producing the final state string.
# "clockwise" matches the current angle-sorting in order_face_stickers.
RING_DIRECTION = "clockwise"

# Optional: neighbor layout (not required for the current state string output).
FACE_NEIGHBORS = {
    "black": ["light_blue", "light_yellow", "light_orange", "light_green", "light_red"],
    "light_yellow": ["dark_red", "light_orange", "black", "light_blue", "dark_green"],
    "light_orange": ["black", "light_yellow", "dark_red", "dark_blue", "light_green"],
    "light_green": ["dark_blue", "dark_yellow", "light_red", "black", "light_orange"],
    "light_red": ["black", "light_green", "dark_yellow", "dark_orange", "light_blue"],
    "light_blue": ["black", "light_red", "dark_orange", "dark_green", "light_yellow"],

    "dark_green": ["white", "dark_red", "light_yellow", "light_blue", "dark_orange"],
    "dark_red": ["white", "dark_blue", "light_orange", "light_yellow", "dark_green"],
    "dark_blue": ["light_orange", "dark_red", "white", "dark_yellow", "light_green"],
    "dark_yellow": ["white", "dark_orange", "light_red", "light_green", "dark_blue"],
    "dark_orange": ["white", "dark_green", "light_blue", "light_red", "dark_yellow"],
    "white": ["dark_green", "dark_orange", "dark_yellow", "dark_blue", "dark_red"],
}

LABEL_TO_COLOR = {
    "light_yellow": (241, 233, 151),
    "dark_yellow":  (240, 232, 66),
    "light_blue":   (148, 195, 236),
    "dark_blue":    (7, 98, 206),
    "light_orange": (242, 162, 129),
    "dark_orange":  (244, 147, 60),
    "light_red":    (234, 199, 190),
    "dark_red":     (228, 92, 142),
    "black":        (25, 75, 181),
    "white":        (226, 232, 233),
    "light_green":  (188, 217, 192),
    "dark_green":   (147, 211, 94),
}
