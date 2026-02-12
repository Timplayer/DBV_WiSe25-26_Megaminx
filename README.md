# DBV WiSe25/26 Megaminx

Dieses Projekt erkennt die Sticker-Farben eines Megaminx aus Bildern, ordnet sie den 12 Farb-Labels zu und visualisiert das Ergebnis als entfaltetes Netz des Dodekaeders.

## Features

- Konturerkennung und Face-Center-Suche mit OpenCV
- Sticker-Erkennung pro Face inkl. Winkel-basierter Sortierung
- Farbklassifikation per `kNN` (scikit-learn) auf Basis von `data/colors.csv`
- Voting ueber mehrere Bilder fuer robustere Endzuordnung
- Visualisierung als Dodekaeder-Netz

## Projektstruktur

```text
.
|-- data/
|   |-- colors.csv
|   |-- normal_test_data/
|   |-- high_light_test_data/
|   |-- 1_face/
|   |-- 3_faces/
|   `-- faces/
|-- src/
|   |-- megaminx_detection.py
|   |-- color_mapping.py
|   |-- canonical.py
|   `-- visualize.py
|-- requirements.txt
`-- README.md
```

## Voraussetzungen

- Python 3.10+ (empfohlen: 3.11)
- OpenCV-faehige Umgebung mit GUI-Unterstuetzung (`cv2.imshow`)

## Installation

```bash
pip install -r requirements.txt
```

## Ausfuehrung

Wichtig: Das Hauptskript nutzt relative Pfade und sollte aus dem Ordner `src/` gestartet werden.

```bash
cd src
python megaminx_detection.py
```

Standardmaessig werden Bilder aus `../data/normal_test_data` geladen (siehe `IMAGE_DIR` in `src/megaminx_detection.py`).

## Pipeline (kurz)

1. Kanten/Masken aus Bild extrahieren (`find_contours`)
2. Face-Center aus Konturen bestimmen (`find_face_centers`)
3. Sticker je Face finden (`find_sticker_for_face`)
4. Sticker-Reihenfolge ueber Winkel und Nachbarschaft normalisieren (`find_index_of_stickers`)
5. Ueber mehrere Bilder voten (`vote_faces`)
6. Ergebnis visualisieren (`visualize`)

## Datenbasis fuer Farberkennung

- Trainingsdaten liegen in `data/colors.csv`.
- Jede Spalte repraesentiert ein Farb-Label (z. B. `light_blue`, `dark_red`, `white`).
- Die Klassifikation erfolgt in `src/color_mapping.py` ueber ein skaliertes `kNN`-Modell.

## Konfiguration

Relevante Konstanten in `src/megaminx_detection.py`:

- `MAX_FACES = 3`: maximal erkannte Faces pro Bild
- `CENTER_RADIUS = 6`: Sampling-Radius fuer Farbmessung
- `DEBUG = False`: Debug-Fenster aktivieren/deaktivieren

## Hinweise und bekannte Einschraenkungen

- Aktuell ist die Verarbeitung auf bis zu 3 sichtbare Faces pro Bild ausgelegt.
- Schlechte Beleuchtung/Reflexionen koennen Konturerkennung und Farbmapping verschlechtern.
- In Headless-Umgebungen (ohne Display) schlagen Visualisierungsfenster fehl.
