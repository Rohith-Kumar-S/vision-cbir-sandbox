# Content-Based Image Retrieval System (CBIR)

A **Content-Based Image Retrieval (CBIR)** system that combines **deep learning features (ResNet)** with **traditional computer vision techniques** such as **color histograms, gradient textures, and Laws filters** to retrieve visually similar images.

The system allows users to **interactively adjust feature weights through a GUI** to prioritize **color, texture, or semantic similarity**, enabling customized retrieval results.

---

# Features

## Deep Learning Features

| Feature Type | Description | Dimensions | Distance |
|---------------|-------------|------------|-----------|
| **ResNet18** | Semantic similarity using **ResNet-18** embeddings pre-trained on ImageNet | 512-d | Cosine similarity |

---

## Color Features

| Feature Type | Description | Dimensions | Distance |
|---------------|-------------|------------|-----------|
| **Center Patch** | 7×7 center patch extracted from image | 49-d (7×7) | L2 |
| **rg** | Illumination-invariant color histogram using normalized r,g values | 256-d (16×16) | Histogram Intersection |
| **rgb** | Full RGB color distribution capturing color correlations | 512-d (8×8×8) | Histogram Intersection |
| **multi-rg** | rg histograms computed on **top and bottom halves** of the image | 256-d + 256-d | Histogram Intersection |

---

## Texture Features

| Feature Type | Description | Dimensions | Distance |
|---------------|-------------|------------|-----------|
| **texture-grads** | Gradient-based texture using magnitude and orientation histograms | 256-d (16×16) | L1 |
| **texture-s5s5** | Laws filter response (S5S5) for blob/spot detection | 16-d | L1 |

---

# Distance Metrics

| Metric | Used For |
|------|-----------|
| **Cosine Distance** | Deep learning embeddings |
| **Histogram Intersection** | Color histograms |
| **L1 Distance** | Gradient texture features |
| **L2 Distance** | Center patch features |

---

# GUI Controls

## Feature Selection

### Color Feature Types
- **Center Patch** — small region extracted from image center  
- **rg** — rg chromaticity histogram  
- **Multi-rg** — multiple rg histograms from different image regions  

### Texture Feature Types
- **Magnitude + Orientation** — gradient-based texture descriptor  
- **Texture + Color** — combined feature approach  
- **Laws Filters** — S5S5 filter responses  

### DNN Feature Types
- **ResNet** — ResNet-18 embeddings for semantic similarity  

---

## Weight Adjustment

| Control | Range | Purpose |
|-------|------|--------|
| **Color Feature Weight** | 0.0 – 1.0 | Importance of color similarity |
| **Texture Feature Weight** | 0.0 – 1.0 | Importance of texture patterns |
| **DNN Feature Weight** | 0.0 – 1.0 | Importance of semantic similarity |

---

## Retrieval Parameters

| Parameter | Description |
|----------|-------------|
| **Number of Results (K)** | Number of top similar images to retrieve |

---

## Buttons

| Button | Function |
|------|-----------|
| **Select Query Image** | Choose an image from the file system |
| **Search Similar Images** | Execute retrieval with current feature settings |

---

# Results Display

Retrieved images are displayed in a **scrollable 3-column grid**.

Each result shows:

- **Thumbnail image** (150 × 150)
- **Filename**
- **Distance score** *(lower = more similar)*

---

# Requirements

### Python Implementation

- Python **3.8+**
- OpenCV (`opencv-python`)
- NumPy
- Pillow (`PIL`)
- `tkinter` *(usually included with Python)*

### C++ Implementation

- **OpenCV 4.x**
- **C++17 or later**
- **CMake 3.10+**

---

# Installation

### Python

```bash
pip install opencv-python numpy pillow# vision-recognition-sandbox