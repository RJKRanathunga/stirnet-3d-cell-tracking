# BioHub Cell Detection & Tracking Project
**Project Status:** In Progress  
**Last Updated:** 2026-07-27

---

# Current Progress

The current pipeline has been implemented successfully up to **cell tracking**.

Current pipeline:

```
Raw 3D Microscopy Images
        │
        ▼
Preprocessing
        │
        ▼
Binary Mask Generation
        │
        ▼
Watershed Segmentation
        │
        ▼
Cell Feature Extraction
        │
        ▼
Cell Tracking
```

The overall pipeline is functional and capable of detecting and tracking a large number of cells throughout the time sequence.

---

# Current Bottleneck

The primary limitation of the entire pipeline is **cell detection**, not the tracking algorithm itself.

## Problem

When two (or more) cells move very close together, the binary mask merges them into a single connected component.

As a result:

- Watershed receives a merged region.
- The merged region is detected as a single cell.
- Only one centroid is produced.
- The tracker receives incorrect detections.
- Tracking errors propagate through later time points.

Therefore, improving **cell separation** is currently the highest priority.

---

# Current Observations

Several observations have been made after analyzing the tracking results.

## 1. Cell Births and Deaths

A large number of tracks appear to be created and destroyed throughout a sequence.

However, most of these are **not actual biological events**.

The majority occur because cells:

- enter the imaging volume,
- leave the imaging volume,
- partially appear near image boundaries.

These boundary events create false births and deaths.

This noise can later be reduced using boundary filtering and additional post-processing.

---

## 2. Remaining Tracking Errors

After ignoring boundary events, the majority of remaining tracking failures originate from a single source:

> Incorrect cell detection caused by merged cells.

This confirms that improving segmentation quality will have a much larger impact than modifying the tracker itself.

---

# Current Development Strategy

Rather than replacing the existing segmentation pipeline, the goal is to **build an additional refinement stage** on top of the current algorithm.

Current pipeline:

```
Binary Mask
      │
      ▼
Watershed
      │
      ▼
Detected Cells
```

Proposed pipeline:

```
Binary Mask
      │
      ▼
Watershed
      │
      ▼
Candidate Cell Detection
      │
      ▼
Merged Cell Refinement (New Stage)
      │
      ▼
Final Cell Detection
      │
      ▼
Tracking
```

The current watershed-based approach already performs well for the majority of cells.

The new refinement stage should only operate on difficult regions where multiple touching cells have been merged.

This approach minimizes unnecessary modifications while preserving the strengths of the existing pipeline.

---

# Current Research Direction

The refinement algorithm will exploit image information that still exists in the original microscopy images but is lost in the binary mask.

Potential cues include:

- intensity valleys between touching cells,
- local intensity gradients,
- changes in cross-sectional area across Z slices,
- variation of intensity through the Z dimension,
- morphological changes of connected regions,
- other image-processing techniques that can reveal hidden boundaries.

The objective is to separate merged cells without replacing the existing segmentation pipeline.

---

# Current Task

Before developing a better separation algorithm, a dataset of failure cases is required.

The algorithm must be evaluated specifically on the situations where it currently fails.

Therefore, the current task is **collecting merged-cell examples**.

---

# Visualization Progress

A visualization notebook has already been extended to display:

- cell IDs
- segmentation
- tracks

Each detected cell is now labeled with its corresponding **cell ID**, making it much easier to identify problematic detections.

---

# Next Implementation

The next tool to implement is a **cell volume extraction utility**.

Input:

- Cell ID
- Time point

Output:

A cubic 3D volume centered around that detected cell.

Example:

```
Input:
    Cell ID = 153
    Time = 42

↓

Locate centroid

↓

Extract nearby 3D cube

↓

Save cropped image volume
```

This extracted volume will allow detailed inspection of difficult cases.

---

# Purpose of Volume Extraction

The extracted volumes will serve several purposes:

- identify exactly why the algorithm failed,
- inspect touching cells,
- analyze original intensity distributions,
- compare successful and failed detections,
- evaluate future segmentation improvements,
- build a collection of representative failure cases.

This dataset will become the benchmark for developing and validating the new refinement algorithm.

---

# Future Roadmap

## Immediate Next Step

- Implement 3D volume extraction using Cell ID and time point.
- Collect a library of merged-cell examples.
- Analyze failure patterns.
- Design and evaluate improved merged-cell separation methods.

---

## After Cell Separation Improves

Once cell detection becomes sufficiently accurate:

- improve cell tracking consistency,
- reduce fragmented tracks,
- implement robust track stitching,
- improve birth/death event handling,
- implement cell division (mitosis) detection,
- further refine lineage reconstruction.

---

# Current Project State Summary

## Completed

- ✅ Image preprocessing
- ✅ Binary mask generation
- ✅ Watershed segmentation
- ✅ Cell feature extraction
- ✅ Cell tracking
- ✅ Track visualization
- ✅ Cell ID visualization

---

## In Progress

- 🔄 Collect merged-cell failure examples
- 🔄 Implement 3D cell volume extraction
- 🔄 Research merged-cell separation algorithms

---

## Planned

- ⏳ Merged-cell refinement stage
- ⏳ Improved tracking
- ⏳ Boundary event filtering
- ⏳ Cell birth/death handling
- ⏳ Cell division detection
- ⏳ Cell lineage reconstruction

---

# Overall Project Status

The pipeline is already capable of detecting and tracking cells across time.

The current challenge is no longer building the pipeline itself, but improving **cell detection accuracy in difficult cases where neighboring cells become merged**.

Rather than replacing the existing segmentation pipeline, the current strategy is to develop a targeted refinement stage that resolves merged cells while preserving the strengths of the current watershed-based approach.

Once this bottleneck is addressed, the remaining tracking and lineage reconstruction tasks can be built on a much more reliable set of cell detections.