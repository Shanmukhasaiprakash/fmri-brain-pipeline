# 🧠 fMRI Brain Image Processing Pipeline
### Preprocessing · GLM Activation · ROI Identification · Functional Connectivity · Visualization

[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b%2B-orange?logo=mathworks)](https://www.mathworks.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Data-OpenNeuro%20ds000114-green)](https://openneuro.org/datasets/ds000114)
[![Institution](https://img.shields.io/badge/Rutgers-BINF%207550-CC0000)](https://shp.rutgers.edu)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

> **A complete, toolbox-independent MATLAB pipeline for functional MRI preprocessing, statistical activation mapping, functional connectivity analysis, and multi-dimensional visualization — validated on real 3T scanner data.**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Results](#results)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Methods Summary](#methods-summary)
- [Authors](#authors)
- [Citation](#citation)

---

## Overview

This project implements a six-stage fMRI image processing pipeline in a **single MATLAB script** (`Project.m`), requiring no external neuroimaging toolboxes (no SPM, FSL, or FreeSurfer). The pipeline was developed as part of **BINF 7550 — Medical Image Processing and Visualization** at Rutgers School of Health Professions, and validated on the publicly available OpenNeuro ds000114 finger/foot/lips motor task dataset.

**Key results on real 3T data:**
- ✅ `510 significant voxels` detected (p < 0.001, uncorrected)
- ✅ `Peak t = 5.13` in bilateral motor network
- ✅ `10 ROIs` identified including L/R Motor Cortex, Sensory Cortex, Cerebellum, Prefrontal
- ✅ `10×10 functional connectivity matrix` with strong bilateral motor coupling (Fisher z ≈ 0.94)
- ✅ `Mean tSNR = 41.4` confirming adequate data quality

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    fMRI PROCESSING PIPELINE                              │
├──────────┬──────────────┬──────────────┬──────────┬──────────┬──────────┤
│ Stage 1  │   Stage 2    │   Stage 3    │ Stage 4  │ Stage 5  │ Stage 6  │
│  NIfTI   │ Preprocessing│  Feature     │   ROI    │   FC     │  Visual  │
│  Load    │              │  Extraction  │  Detect  │ Analysis │   ize    │
├──────────┼──────────────┼──────────────┼──────────┼──────────┼──────────┤
│ Header   │ Motion       │ tSNR map     │ t > 3.1  │ Pearson  │ Hist EQ  │
│ parse    │ correction   │              │ threshold│    r     │ Edge det │
│          │ (phase corr) │ GLM + HRF    │          │          │ Morphol  │
│ Voxel    │ FFT high-    │ convolution  │ Connected│ Fisher   │ ops      │
│ data     │ pass filter  │              │ component│ z-score  │          │
│          │              │ t-statistic  │ labeling │          │ GIF      │
│ Brain    │ Gaussian     │ map          │          │ FC       │ anim-    │
│ mask     │ smoothing    │              │ ROI      │ matrix   │ ation    │
│          │ FWHM=6mm     │              │ labels   │          │          │
└──────────┴──────────────┴──────────────┴──────────┴──────────┴──────────┘
```

---

## Results

### Preprocessing QC
Real brain anatomy visible — bilateral motor strip, cortical ribbon, brainstem cross-section.
Mean tSNR = 35 | Max = 192.9 | Mean FD = 0.000 mm

### GLM Activation (p < 0.001)
| ROI | Label | Voxels | Peak t |
|-----|-------|--------|--------|
| 1 | Cerebellum | 26 | 2.76 |
| 2 | Prefrontal Cortex | 87 | **5.13** |
| 3 | Prefrontal Cortex | 52 | 4.56 |
| 7 | L Sensory Cortex | 18 | 3.47 |
| 8 | L Motor Cortex | 46 | 3.18 |
| 9 | R Motor Cortex | 91 | 4.62 |
| 10 | L Motor Cortex | 11 | 3.79 |

### Functional Connectivity
- **L Motor ↔ R Motor**: Fisher z ≈ 0.87–0.98 (bilateral motor network)
- **M1 ↔ S1**: Fisher z ≈ 1.08 (sensorimotor integration)
- **PFC ↔ PFC**: Fisher z ≈ 0.81–1.24 (motor planning network)

---

## Dataset

**OpenNeuro ds000114** — A test-retest fMRI dataset for motor, language and spatial attention functions

| Parameter | Value |
|-----------|-------|
| Scanner | Siemens 3T Trio TIM |
| Sequence | T2*-weighted EPI |
| Matrix | 64 × 64 × 30 voxels |
| TR | 2.5 s |
| Volumes | 184 (7.7 min) |
| Task | Finger/Foot/Lips — 15 blocks × 15s |
| License | CC0 (public domain) |
| DOI | [10.18112/openneuro.ds000114.v1.0.2](https://doi.org/10.18112/openneuro.ds000114.v1.0.2) |

---

## How to Run

### Requirements
- MATLAB R2022b or newer
- Image Processing Toolbox (for `histeq`, `edge`, `strel`, `imerode`, `imdilate`, `imopen`, `bwconncomp`)
- No other toolboxes required

### Steps

**1. Download the dataset**
```
OpenNeuro ds000114 → sub-01 → ses-test → func →
sub-01_ses-test_task-fingerfootlips_bold.nii.gz
```
Also download: `task-fingerfootlips_events.tsv` (root level)

**2. Unzip the NIfTI file**
```matlab
gunzip('path\to\sub-01_ses-test_task-fingerfootlips_bold.nii.gz')
```

**3. Set your data path in Project.m (lines 18–19)**
```matlab
NIFTI_PATH = 'C:\path\to\8_finger_foot_lips.nii';
```
The events TSV file will be found automatically if placed in the same folder.

**4. Run**
```matlab
Project
```

That's it — all outputs save to a `results/` folder automatically.

---

## Output Files

| File | Description |
|------|-------------|
| `01_preprocessing_qc.png` | Motion params, FD, tSNR distribution + axial slice maps |
| `02_image_enhancement.png` | Histogram EQ, Canny/Sobel edge detection, morphological ops |
| `03_activation_maps.png` | GLM t-map overlaid on anatomy (axial, coronal, sagittal) |
| `04_connectivity.png` | 10×10 FC matrix + weighted network graph |
| `brain_activity_animation.gif` | 30-slice animated brain activity movie |

---

## Methods Summary

| Stage | Technique | MATLAB Function |
|-------|-----------|-----------------|
| Motion correction | Phase-correlation translation | `fft2`, `ifft2` |
| Temporal filtering | FFT-based high-pass (>128s) | `fft`, `ifft` |
| Spatial smoothing | 3D Gaussian (FWHM=6mm) | `convn` |
| Temporal SNR | Mean/std voxelwise | `mean`, `std` |
| GLM | Pseudoinverse estimation | `pinv` |
| ROI clustering | 6-connectivity labeling | `bwconncomp` |
| Connectivity | Pearson r + Fisher z | matrix operations |
| Histogram EQ | Intensity redistribution | `histeq` |
| Edge detection | Canny + Sobel | `edge` |
| Morphology | Erosion, dilation, opening | `imerode`, `imdilate`, `imopen` |
| Animation | Frame-by-frame GIF | `rgb2ind`, `imwrite` |

---

## Authors

**Shanmukha Sai Prakash Jeelakarra**
- Department of Health Informatics, Rutgers School of Health Professions
- [LinkedIn](https://linkedin.com/in/shanmukhasaiprakash) | [GitHub](https://github.com/shanmukhasaiprakash)

**Dhana Lakshmi Kakumanu**
- Department of Health Informatics, Rutgers School of Health Professions
- [LinkedIn](https://linkedin.com/in/dhanalakshmikakumanu) | [GitHub](https://github.com/dhanalakshmikakumanu)

**Course:** BINF 7550 — Medical Image Processing and Visualization
**Instructor:** Dr. Dinesh P. Mital
**Institution:** Rutgers School of Health Professions, Newark NJ

---

## Citation

If you use this pipeline, please cite:

```bibtex
@misc{jeelakarra2026fmri,
  title     = {Preprocessing and Visualization of fMRI Brain Images for
               Identification and Highlighting of Regions of Interest},
  author    = {Jeelakarra, Shanmukha Sai Prakash and Kakumanu, Dhana Lakshmi},
  year      = {2026},
  note      = {BINF 7550 course project, Rutgers School of Health Professions},
  url       = {https://github.com/your-username/fmri-pipeline}
}
```

**Dataset:**
> Gorgolewski KJ, Storkey A, Bastin ME, Whittle IR, Wardlaw JM, Pernet CR (2022).
> A test-retest fMRI dataset for motor, language and spatial attention functions.
> OpenNeuro. https://doi.org/10.18112/openneuro.ds000114.v1.0.2

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Built with MATLAB · Validated on OpenNeuro ds000114 · Rutgers SHRP 2026*

