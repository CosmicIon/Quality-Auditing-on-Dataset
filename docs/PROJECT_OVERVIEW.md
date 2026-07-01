# Project Overview

## Problem Statement
The goal is to develop an automated pipeline that detects label noise, bias, and out-of-distribution (OOD) samples in a specialized dataset, and uses generative models (like Diffusion models for vision) to selectively augment underrepresented classes.

## Major Outcomes Expected
1. **Data Auditing**: Implement cleanlab or embedding-based clustering to automatically flag mislabeled or highly ambiguous samples in a public benchmark dataset.
2. **Controlled Synthesis**: Use Conditional GANs/Diffusion Models (for vision) to generate synthetic samples exclusively for the identified weak clusters or minority classes.
3. **Fidelity & Diversity Metrics**: Evaluate the synthetic data using metrics like Fréchet Inception Distance (FID) for images to ensure quality.
4. **Downstream Validation**: Train a baseline classifier on the original data versus the cleaned/augmented data to prove that data-centric interventions yield better accuracy gains than architectural changes.

## Dataset Focus: CIFAR-10
While CIFAR-10 is a standard benchmark, in real-world scenarios or modified versions of such datasets, several major issues arise with images that this pipeline aims to solve:
1. **Wrong Labels**: Images assigned to the incorrect class.
2. **Confusing Images**: Blurry images, images containing multiple objects (e.g., both a cat and a dog), or images where humans cannot agree on the correct label.
3. **Distribution/Imbalance**: Skewed class distributions (e.g., 5000 dogs, 5000 cats, but only 200 rabbits).
4. **Outliers**: Completely irrelevant out-of-distribution samples (e.g., cars in a dataset intended for animals).
