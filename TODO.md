# Project TODOs

## Phase 1: Setup & Data Preparation
- [x] Initialize Python project and environment and make proper folder structure.
- [x] Download and load the CIFAR-10 dataset.
- [x] Explore data structure and set up initial data loaders.

## Phase 2: Data Auditing (Task 1)
- [x] Implement `cleanlab` or custom embedding-based clustering.
- [x] Create a script to scan the dataset and flag issues:
  - [x] Incorrect labels.
  - [x] Confusing/ambiguous images.
  - [x] Outliers (e.g., non-animal/vehicle images in specific classes).
  - [x] Class imbalance.
- [x] Generate an automated auditing report based on the findings.

## Phase 3: Controlled Synthesis (Task 2)
- [x] Select and set up a generative model (Conditional GANs or Diffusion Models).
- [x] Train the generative model using the cleaned subsets of the data.
- [x] Selectively generate synthetic samples for underrepresented/minority classes to balance the dataset.

## Phase 4: Quality Assessment (Task 3)
- [x] Implement Fréchet Inception Distance (FID) metrics for evaluation.
- [x] Evaluate the generated images for realism and diversity.
- [x] Ensure synthetic samples are not exact duplicates (memorization check) of training data.
- [x] Ensure generated data distribution resembles the real data distribution.

## Phase 5: Downstream Validation (Task 4)
- [ ] Define and train a baseline classifier on the **original** CIFAR-10 dataset.
- [ ] Train the same classifier architecture on the **cleaned** dataset (post-auditing).
- [ ] Train the same classifier architecture on the **cleaned and augmented** dataset (post-synthesis).
- [ ] Compare validation/test accuracies across all three experiments.
- [ ] Generate final report proving the effectiveness of the data-centric interventions.
