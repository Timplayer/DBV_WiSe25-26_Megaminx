RING_DIRECTION = "clockwise"

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
    "light_yellow": (151, 233, 241),
    "dark_yellow":  (66, 232, 240),
    "light_blue":   (236, 195, 148),
    "dark_blue":    (206, 98, 7),
    "light_orange": (129, 162, 242),
    "dark_orange":  (60, 147, 244),
    "light_red":    (190, 199, 234),
    "dark_red":     (142, 92, 228),
    "black":        (181, 75, 25),
    "white":        (233, 232, 226),
    "light_green":  (192, 217, 188),
    "dark_green":   (94, 211, 147),
}
