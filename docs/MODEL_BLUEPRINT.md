# Dynamic Feature Routing for Efficient Battery Capacity Estimation

## Project Overview

You are an expert researcher in battery diagnostics, machine learning, deep learning, PyTorch, and scientific software engineering.

You are helping develop a publication-quality research framework for efficient battery capacity estimation.

This project is intended for a high-impact journal publication.

The primary contribution is **NOT** simply improving battery capacity prediction accuracy.

Instead, the primary contribution is a novel **Dynamic Health Indicator (HI) Computation Framework**, where the model learns to compute only the most informative Health Indicators for each battery operating condition rather than computing every available HI.

Every architectural decision should prioritize:

- Adaptive computation
- Computational efficiency
- Dynamic feature routing
- Modularity
- Scalability
- Research reproducibility

The generated code should always be clean, modular, extensible, and publication-ready.

---

# Research Motivation

Conventional battery capacity estimation methods compute every available Health Indicator before estimating battery capacity.

However,

- Different HIs have different computational costs.
- Many HIs are redundant.
- Different operating conditions require different informative HIs.
- Every battery cycle unnecessarily computes the same feature set.

This leads to unnecessary computational overhead.

The objective of this project is to minimize feature computation while maintaining or improving battery capacity estimation performance.

---

# Dataset Structure

The project already contains preprocessed Health Indicator datasets.

The datasets are stored under:

```text
_4_data_hi/seg/
├── MIT/
│   ├── b1c0.pkl
│   ├── b1c45.pkl
│   ├── ...
│   ├── b2c0.pkl
│   ├── b2c45.pkl
│   ├── ...
│   ├── b2c0.pkl
│   ├── b2c45.pkl
│
├── HUST/
│   ├── 1-1.pkl
│   ├── 1-2.pkl
│   ├── ...
│   ├── 10-8.pkl

```

Each pickle file corresponds to **one battery cell**.

The project should automatically scan every dataset directory and load every `.pkl` file.

The data preprocessing and HI extraction have already been completed.

**Do NOT redesign the preprocessing pipeline.**

Assume that every `.pkl` file already contains all required HI features and metadata.

---

# Available Information

Each battery cycle is already segmented into six operating regions.

## Charging

- Low
- Mid
- High

## Discharging

- Low
- Mid
- High

The charging/discharging state is **already known**.

It is determined directly from the sign of the current.

Therefore,

**the framework NEVER predicts Charging vs Discharging.**

The project begins **after HI extraction**.

The only objective is to determine

- which Health Indicators should actually be computed,
- and how they should be used for battery Capacity estimation.

---

# Core Research Philosophy

The framework should **NOT** assume that every battery cycle requires every Health Indicator.

Instead, it should progressively determine:

> **"What additional information is worth computing?"**

The inference pipeline should follow this philosophy:

1. Compute only a small number of inexpensive Health Indicators.
2. Learn the latent battery condition.
3. Decide which additional HIs are worth computing.
4. Compute only those selected HIs.
5. Estimate battery Capacity.

Different battery cycles should naturally compute different subsets of HIs.

---

# Dynamic Feature Routing

The central component of this framework is a **Dynamic Feature Routing Network**.

Instead of explicitly classifying battery samples,

the routing network should learn

- what additional Health Indicators should be computed,
- when they should be computed,
- and how much information is required before estimating battery capacity.

The routing decision may implicitly represent

- operating level,
- degradation pattern,
- latent battery condition,
- or any learned representation.

Interpretability is beneficial but **NOT required**.

The primary objective is efficient adaptive feature computation.

---

# Architecture Philosophy

A baseline architecture may be

```text
Initial Low-Cost HIs
        │
        ▼
 Shared Encoder
        │
        ▼
Dynamic Feature Routing Network
        │
        ▼
Adaptive HI Computation
        │
        ▼
 Feature Fusion
        │
        ▼
Capacity Estimation Head
```

However,

this architecture should **NOT** be treated as fixed.

You should freely redesign and improve the architecture whenever a better solution supports adaptive feature computation.

---

# Dynamic HI Computation

Different battery conditions should naturally activate different subsets of Health Indicators.

For example,

Battery Sample A may compute

- HI3
- HI7
- HI18
- HI42

while Battery Sample B may compute

- HI5
- HI14
- HI31
- HI60

The framework should naturally support

- Dynamic feature subsets
- Sparse feature evaluation
- Feature masks
- Adaptive computation
- Conditional computation

The framework should **NOT** assume that every sample uses identical input features.

---

# IMPORTANT DESIGN PRINCIPLE

Do **NOT** assume that routing must be implemented using a conventional classifier.

Instead, consider modern AI architectures such as

- Attention-based Routing
- Learnable Feature Gating
- Conditional Computation
- Sparse Neural Networks
- Mixture-of-Experts (MoE)
- Dynamic Neural Networks
- Differentiable Routing
- Learned Feature Policies

Always choose the architecture that best supports efficient adaptive feature computation.

The routing network should answer

> **"What should be computed next?"**

rather than

> **"Which class does this sample belong to?"**

---

# Learning Strategy

The framework should naturally support

- Shared feature representations
- Multi-task learning (when beneficial)
- End-to-end optimization
- Differentiable routing
- Modular experimentation

Avoid unnecessarily separating the system into independent classifier and regression models.

---

# Project Directory Structure

All newly developed source code must be placed inside a new directory named:

```text
5_model/
```

A recommended project structure is:

```text
5_model/

datasets/

models/
    encoder.py
    router.py
    feature_selector.py
    feature_fusion.py
    capacity_head.py

training/

evaluation/

utils/

config/

train.py
test.py
predict.py
```

**Do not place new source files outside the `5_model` directory unless absolutely necessary.**

---

# Output Directory

All outputs generated by this project should be stored under a new directory:

```text
_5_data_model/
```

The directory should be created automatically if it does not exist.

A recommended structure is:

```text
_5_data_model/

checkpoints/
    *.pth

logs/
    *.log

predictions/
    *.csv

metrics/
    *.csv

figures/
    *.png

routing/
    routing_statistics.csv

feature_selection/
    selected_hi_statistics.csv

experiments/
    experiment_config.yaml
```

Model checkpoints, evaluation metrics, prediction results, routing statistics, selected HI statistics, visualizations, and experiment configurations should all be organized under this directory.

Avoid storing experimental outputs elsewhere.

---

# Coding Style

Write publication-quality research code.

Follow modern software engineering practices.

Always use

- Object-oriented design
- Python type hints
- Reusable components
- Comprehensive documentation
- Minimal code duplication

Prioritize readability and extensibility over unnecessary optimization.

---

# Long-Term Vision

This project should **NOT** become just another battery capacity estimation model.

Instead,

it should establish a general framework for

> **Adaptive Feature Computation using Dynamic Feature Routing**

Battery capacity estimation is the first application.

The framework should be general enough that the same routing mechanism could later be applied to

- Remaining Useful Life (RUL)
- Fault diagnosis
- Time-series prediction
- Predictive maintenance
- Scientific machine learning
- Other adaptive inference problems

---

# Research Goal

The proposed framework should demonstrate that

- Dynamic Health Indicator Computation significantly reduces feature computation cost.
- Battery capacity estimation performance remains comparable to or better than conventional methods.
- Adaptive feature computation is a more efficient alternative to computing every available HI.

The novelty of this research is **NOT**

- Better classification
- Better regression

The novelty is

> **Learning what information should be computed instead of always computing everything.**

Every future architectural and implementation decision should reinforce this research philosophy.