# Automated Pipeline Tasks

The system acts as an automated assistant/agent that performs the following major tasks sequentially:

## Task 1: Inspect the Dataset (Data Auditing)
Examine every sample in the dataset to identify issues.
- **Checks Performed**:
  - Does the label seem correct?
  - Is the sample confusing or ambiguous?
  - Does it look significantly different from others (outlier)?
  - Is one class significantly smaller than others (imbalance)?
- **Output**: Generates a comprehensive auditing report.
  - *Example Report:* 
    - "I found 300 suspicious images."
    - "Rabbit class has very few examples."
    - "100 images may be mislabeled."

## Task 2: Repair the Dataset (Controlled Synthesis)
After identifying the problems, the system improves the dataset.
- **Action**: Instead of creating random new images across all classes, the system selectively creates *only* the images that are actually needed (e.g., to balance minority classes or replace heavily corrupted images).
- **Methods**: Conditional GANs or Diffusion Models.

## Task 3: Check Whether the New Data is Good (Quality Assessment)
Creating new images is easy; creating useful, high-quality images is difficult.
- **Checks Performed**:
  - Do the new images look realistic?
  - Are they diverse and different from one another?
  - Are they simply memorized copies of the training data?
  - Do they resemble the real data distribution?
- **Output**: The system automatically evaluates the quality of the generated data using metrics like Fréchet Inception Distance (FID).

## Task 4: Prove That Things Improved (Downstream Validation)
Validate the effectiveness of the data-centric pipeline by comparing downstream model performance.
- **Experiments**:
  1. Train a baseline model using the **original dataset**.
  2. Train the model after **cleaning** the dataset (removing/fixing bad samples).
  3. Train the model after **cleaning and adding new synthetic samples**.
- **Success Criteria**: Accuracy after cleaning and targeted augmentation should be demonstrably higher than training on the original dataset, proving the value of the data-centric approach.
