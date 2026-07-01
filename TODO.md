# Project TODOs

## Phase 1: Setup & Data Preparation
- [x] Initialize Python project and environment and make proper folder structure.
- [x] Download and load the CIFAR-10 dataset.
- [x] Explore data structure and set up initial data loaders.

## Phase 2: Data Auditing (Task 1)
- [ ] Implement `cleanlab` or custom embedding-based clustering.
- [ ] Create a script to scan the dataset and flag issues:
  - [ ] Incorrect labels.
  - [ ] Confusing/ambiguous images.
  - [ ] Outliers (e.g., non-animal/vehicle images in specific classes).
  - [ ] Class imbalance.
- [ ] Generate an automated auditing report based on the findings.

## Phase 3: Controlled Synthesis (Task 2)
- [ ] Select and set up a generative model (Conditional GANs or Diffusion Models).
- [ ] Train the generative model using the cleaned subsets of the data.
- [ ] Selectively generate synthetic samples for underrepresented/minority classes to balance the dataset.

## Phase 4: Quality Assessment (Task 3)
- [ ] Implement Fréchet Inception Distance (FID) metrics for evaluation.
- [ ] Evaluate the generated images for realism and diversity.
- [ ] Ensure synthetic samples are not exact duplicates (memorization check) of training data.
- [ ] Ensure generated data distribution resembles the real data distribution.

## Phase 5: Downstream Validation (Task 4)
- [ ] Define and train a baseline classifier on the **original** CIFAR-10 dataset.
- [ ] Train the same classifier architecture on the **cleaned** dataset (post-auditing).
- [ ] Train the same classifier architecture on the **cleaned and augmented** dataset (post-synthesis).
- [ ] Compare validation/test accuracies across all three experiments.
- [ ] Generate final report proving the effectiveness of the data-centric interventions.
