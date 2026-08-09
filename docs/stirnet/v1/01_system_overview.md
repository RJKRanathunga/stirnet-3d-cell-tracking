# STIRNET — Spatio-Temporal Instance Refinement Network

# 01 — System Overview

## 1. Purpose

STIR-Net V1 is a **cell-instance correction model**. It consumes an imperfect segmentation together with dense image evidence and provisional temporal clues and produces a corrected segmentation for the target frame.

It is not a tracker.

It is not a replacement for the existing segmentation pipeline.

It is not a generic full-volume instance-segmentation model trained to discover every cell from scratch.

Its role is:

```text
reasonable initial segmentation
        +
raw geometric evidence
        +
temporal evidence from surrounding frames
        |
        v
learned instance refinement
        |
        v
more accurate cell instances
```

## 2. Full pipeline position

```text
Stage preprocessing
    |
    v
Existing instance segmentation
    |
    v
Initial cell instances
    |
    v
Trackastra pass 1
    |
    +--> provisional associations
    +--> track starts/ends
    +--> gaps / broken trajectories
    +--> division-related clues
    +--> motion / expected positions
    |
    v
STIR-Net V1
    |
    v
Corrected cell instances
    |
    v
Trackastra pass 2
    |
    v
Final tracks / lineage
```

Trackastra pass 1 is discarded after refinement except for diagnostics. Trackastra pass 2 establishes the final identities and associations from the corrected instance set.

## 3. Target failure modes

The same output formulation must handle all of the following.

### 3.1 Under-segmentation

```text
input:   [      AB      ]
output:  [ A ]     [ B ]
```

One input instance becomes two or more biological cells.

### 3.2 Over-segmentation

```text
input:   [ A1 ] [ A2 ]
output:  [      A      ]
```

Several input fragments become one biological cell.

### 3.3 Missing cell

```text
input:   no instance
output:  [ A ]
```

### 3.4 False positive

```text
input:   [ artifact ]
output:  no cell
```

### 3.5 Boundary correction

```text
input:   approximately correct cell
output:  same cell with corrected boundary
```

### 3.6 Correct cell

```text
input:   [ A ]
output:  [ A ]
```

The model must learn that **no correction is a valid and common outcome**.

## 4. Why not explicit error classifiers?

V1 deliberately avoids primary heads such as:

```text
MERGED / NOT MERGED
MISSING / NOT MISSING
OVERSPLIT / NOT OVERSPLIT
```

The final structural change already identifies the correction type.

Examples:

- 1 input instance → 2 output instances = under-segmentation correction.
- 2 input instances → 1 output instance = over-segmentation correction.
- 0 input instances → 1 output instance = missing-cell recovery.
- 1 input instance → 0 output instances = false-positive removal.

This unifies the learning problem.

## 5. Why Trackastra is used twice

Trackastra pass 1 is useful because temporal continuity can reveal where the current segmentation is inconsistent.

Examples:

```text
t-1       t        t+1
 A ------ ? ------- A
```

or:

```text
A -----\
        [AB]
B -----/
```

The provisional tracking is not assumed to be correct. It is only a source of spatial-temporal hypotheses.

After STIR-Net modifies the instances, the old tracks are no longer valid. Therefore Trackastra is run again from scratch on the corrected instance masks.

## 6. Information domains

V1 contains two principal internal domains.

### Dense spatial domain

Represents:

- raw fluorescence
- foreground
- physical EDT
- current instance boundaries
- current markers
- local morphology
- learned native-grid features

### Sparse object/temporal domain

Represents:

- provisional detections
- tracklet continuity
- starts / ends / gaps
- motion
- spatial neighbours
- division-related clues
- anomaly salience
- reliability

Neither domain is allowed to independently make the final instance decision.

## 7. Joint reasoning principle

The core architecture is:

```text
spatial encoder
      |
      v
spatial features --------------+
                                |
                                v
                         co-reasoning
                                ^
                                |
temporal graph -----------------+
      |
      v
temporal hypotheses
```

Within a co-reasoning block:

1. temporal hypotheses inspect spatial evidence;
2. temporal hypotheses communicate through a hypothesis graph;
3. spatial features read the updated temporal hypotheses;
4. spatial features pass through convolution again.

Thus the final representation encodes both modalities before query-based instance decoding.

## 8. V1 architectural non-goals

V1 does not include:

- global isotropic resampling;
- full-volume transformer attention;
- learned tracking output;
- track identity loss;
- lineage prediction loss;
- deformable 3D convolution;
- deformable 3D cross-attention;
- optical flow;
- adjacent raw-frame CNN input;
- explicit merge classifier;
- explicit missing-cell classifier;
- explicit over-segmentation classifier.

Each can be considered only if V1 evaluation reveals a specific failure that motivates it.
