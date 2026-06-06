# Thumbnail Feature Extraction — Pipeline Summary

## Purpose

This step extracts interpretable, actionable visual features from YouTube thumbnail images. The goal is to characterise thumbnail design choices (face prominence, text usage, colour tone, title–image alignment) in a form that can be used as predictors in engagement regression models. Opaque embeddings were deliberately avoided in favour of named, human-readable features.

## Scope

- **Source videos**: 6,575 videos published from 2022 onward (filtered by `publishedAt >= 2022-01-01` from `yt_videos_with_local_transcripts.parquet`)
- **Input**: thumbnail JPEGs stored in GCS at `gs://socialmediaanalyticsproject/thumbnails/{video_id}.jpg`
- **Output**: `reza/clean_data/thumbnail_features.parquet` — one row per video, 21 columns

---

## Feature Definitions

### Colour
| Column | Type | Description |
|--------|------|-------------|
| `brightness_mean` | float | Mean V channel of HSV image, normalised to [0, 1] |
| `saturation_mean` | float | Mean S channel of HSV image, normalised to [0, 1] |

Computed via fast NumPy — no model.

---

### Face (MediaPipe FaceDetection — model_selection=1, min_confidence=0.4)
| Column | Type | Description |
|--------|------|-------------|
| `face_present` | bool | At least one face detected |
| `face_count` | int | Total faces detected |
| `face_area_ratio` | float | Bounding box area of the largest face ÷ image area |
| `face_position_x` | float | Horizontal centre of largest face [0=left, 1=right] |
| `face_position_y` | float | Vertical centre of largest face [0=top, 1=bottom] |
| `face_center_distance` | float | Euclidean distance of face centre from image centre |

**Dataset rates**: 82.4% of thumbnails contain at least one face; 23.6% of those contain multiple faces. Median face area ratio is 0.029 (faces occupy ~3% of the image area on average).

---

### Face Pose & Expression (MediaPipe FaceMesh — 468+10 landmarks, refine_landmarks=True)
| Column | Type | Description |
|--------|------|-------------|
| `face_yaw_deg` | float | Left–right head rotation (degrees); negative = turned left |
| `face_pitch_deg` | float | Up–down head tilt (degrees); negative = tilted down |
| `face_roll_deg` | float | In-plane rotation (degrees) |
| `face_is_frontal` | bool | `True` if `|yaw| < 25°` and `|pitch| < 20°` |
| `mouth_openness` | float | Vertical lip distance ÷ face height ratio |
| `eye_openness_mean` | float | Mean of left and right eye vertical ÷ horizontal aperture |
| `gaze_off_camera` | bool | `True` if mean iris displacement from eye centre > 15% of eye width |

Pose computed via `cv2.solvePnP` against a 6-point 3D face model (ZYX Euler decomposition).

**61.4%** of faces with landmarks are classified as frontal.

#### Known caveats for modelling
- **1,525 rows** (23% of face-present rows) have `face_yaw_deg = face_pitch_deg = 0.0` and `face_is_frontal = False`. This is not a real frontal face — it means FaceMesh failed to extract landmarks (typically because the face is too small: median `face_area_ratio` in this group is ~0.010). Treat these as missing pose data, not as frontal faces.
- `face_roll_deg` is unreliable due to a known `solvePnP` ZYX decomposition ambiguity — values cluster near ±180° for some clearly frontal faces. Use `face_yaw_deg` and `face_pitch_deg` for pose; exclude or bin `face_roll_deg`.
- `gaze_off_camera` returns 0% positives in this dataset — likely due to MediaPipe iris landmark instability on small/compressed thumbnails. Treat with caution or drop.

---

### Text (EasyOCR — Italian + English)
| Column | Type | Description |
|--------|------|-------------|
| `text_present` | bool | At least one text region detected |
| `text_area_ratio` | float | Total bounding box area of all text regions ÷ image area |
| `text_word_count` | int | Total word count across all detected text regions |
| `text_has_number` | bool | `True` if any detected text contains a digit |

**72.9%** of thumbnails contain detected text. Font colour and font size were not extracted (EasyOCR does not expose styling metadata).

---

### Cross-modal Alignment (CLIP ViT-L/14)
| Column | Type | Description |
|--------|------|-------------|
| `clip_title_align` | float | Cosine similarity between the CLIP image embedding and the CLIP text embedding of the video title |

Range: [0.049, 0.471], mean 0.249, std 0.066. A higher value means the thumbnail image is more semantically consistent with the title. Full embeddings are not stored — only the scalar similarity.

No null values — all 6,575 rows have a valid score.

---

## Dataset-Level Statistics

| Metric | Value |
|--------|-------|
| Total rows | 6,575 |
| `face_present` rate | 82.4% |
| `text_present` rate | 72.9% |
| `face_is_frontal` rate (face-present rows only) | 61.4% |
| `clip_title_align` mean ± std | 0.249 ± 0.066 |
| Missing values | 0 |

---

## Models and Versions

| Model | Version | Use |
|-------|---------|-----|
| MediaPipe FaceDetection | 0.10.14 | Face presence, count, bounding box |
| MediaPipe FaceMesh | 0.10.14 | Head pose, expression, gaze |
| EasyOCR | 1.7.1 | Text detection and recognition |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 (via transformers) | Title–image alignment |
| OpenCV | 4.9.0.80 | `solvePnP`, HSV colour conversion |

MediaPipe is pinned to 0.10.14 — versions ≥ 0.10.15 restructured the `mp.solutions` API and are incompatible with this codebase.

---

## Infrastructure

- **Extraction**: Google Colab (free tier, T4 GPU). Runtime ~40 minutes for 6,575 images.
- **Checkpointing**: Progress saved to GCS every 500 images — safe to resume after disconnection.
- **Storage**: Output written directly to `gs://socialmediaanalyticsproject/output/thumbnail_features/thumbnail_features.parquet`, then downloaded locally to `reza/clean_data/`.

---

## Recommendations for Modelling

1. **Drop or impute pose columns** (`face_yaw_deg`, `face_pitch_deg`, `face_roll_deg`, `mouth_openness`, `eye_openness_mean`) when `face_present = True` but `face_yaw_deg = 0` — these are missing values, not real measurements. Create a boolean flag `face_pose_available` for the model.
2. **Exclude `face_roll_deg` and `gaze_off_camera`** — unreliable as described above.
3. **`face_area_ratio` and `face_center_distance`** are the most reliable face features across all rows.
4. **`clip_title_align`** is a unique cross-modal feature with no missing values — prioritise it.
5. Consider interaction terms: `face_present × face_is_frontal`, `text_present × text_word_count`, `brightness_mean × saturation_mean`.
