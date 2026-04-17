# Ma'at Submission — CV&IC Project

---

## Motivation *(max 300 words)*

Automatic recognition of incident types from images is relevant to emergency response, disaster management, and social media monitoring. When incidents occur, images are rapidly shared online, and automatically categorising them can accelerate resource allocation and situational awareness for first responders.

The challenge is non-trivial: incident images are visually diverse, often collected under uncontrolled conditions, and exhibit significant class imbalance — a common property of real-world datasets where rare events (e.g. bicycle accidents, nuclear explosions) are underrepresented compared to frequent ones (e.g. car accidents, flooding). Standard classifiers trained on imbalanced data tend to bias toward majority classes, leading to poor recall on the minority classes that may be most critical to detect.

This project is relevant because it addresses both the classification problem and the imbalance problem simultaneously, comparing architectures and balancing strategies in a controlled experimental framework. Transfer learning from ImageNet is the natural approach given the limited dataset size (7,229 images), as it allows leveraging features learned from millions of images.

References: Weber et al. (2020), "Detecting Natural Disasters, Damage, and Incidents in the Wild", ECCV.

---

## (Business/Research) Questions *(max 300 words)*

1. Which pre-trained deep learning architecture — MobileNetV2 or ResNet18 — achieves higher macro-F1 on the Incidents-subset dataset under identical training conditions?

2. Which class imbalance handling strategy — no balancing, class-weighted loss, or hybrid resampling — yields the best macro-F1 across 5-fold cross-validation, specifically for minority classes (< 300 samples)?

3. Does the optimal balancing strategy differ between architectures?

These questions are SMART: they are specific to the dataset and metrics used, measurable through macro-F1 and per-class recall, achievable with the available compute and data, relevant to real-world deployment of incident classifiers, and time-bound to the project scope.

---

## Source Data *(max 500 words)*

The dataset used is a subset of the Incidents1M dataset (Weber et al., 2020), containing 7,229 images across 12 incident categories: airplane accident, bicycle accident, car accident, collapsed, earthquake, flooded, ice storm, nuclear explosion, oil spill, tornado, volcanic eruption, and wildfire.

**Data quality issues considered:**

*Format and corruption:* Image files sourced from the web can be corrupted, truncated, or stored in non-standard formats. We loaded every file with exception handling; 5 files were skipped due to unreadable formats (e.g. hashed desktop files, malformed JPEGs).

*Class imbalance:* The dataset is significantly imbalanced. The largest class (car accident, 947 images) is approximately 4× the size of the smallest (bicycle accident, 228). This is a structural quality issue — a model trained without correction will be biased toward majority classes, reducing macro-F1. We explicitly addressed this through three strategies (see Method).

*Image size heterogeneity:* Raw images vary in resolution and aspect ratio. All images were resized to 224×224 pixels to match the input requirements of the pre-trained models, using PIL with bilinear resampling. This introduces mild distortion for non-square images but ensures consistent input dimensions.

*Mode inconsistency:* Some images were stored in palette mode (mode='P') or RGBA. These were converted to RGB before processing.

**Class distribution:**

| Class | Count | Category |
|---|---|---|
| airplane accident | 854 | Majority |
| bicycle accident | 228 | Minority |
| car accident | 947 | Majority |
| collapsed | 673 | Average |
| earthquake | 906 | Majority |
| flooded | 939 | Majority |
| ice storm | 607 | Average |
| nuclear explosion | 231 | Minority |
| oil spill | 289 | Minority |
| tornado | 272 | Minority |
| volcanic eruption | 618 | Average |
| wildfire | 665 | Average |
| **Total** | **7,229** | |

**Split:** Stratified 5-fold cross-validation, preserving class ratios across folds. No held-out test set was used; all evaluation is on validation folds.

---

## Method *(max 500 words)*

The pipeline consists of five stages applied per fold:

**1. Preprocessing.** Images are resized to 224×224 and normalized using ImageNet channel statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]). This normalization is required because the models were pre-trained on ImageNet — their weights were optimized with inputs at this scale, so matching normalization is essential for effective transfer.

**2. Data augmentation.** Applied to training folds only for generalization: horizontal flip (50%), random rotation (±20°), brightness adjustment (factor 0.8–1.2). Validation folds are never augmented.

**3. Imbalance handling (training fold only).**
- *Strategy A — No balancing:* Training fold used as-is.
- *Strategy B — Class weights:* Per-fold weights computed as w_c = 1/count_c, normalized to sum to the number of classes. Applied to CrossEntropyLoss.
- *Strategy C — Hybrid resampling:* Minority classes oversampled via augmentation; majority classes undersampled to the median class size (~513 images per class per fold, 6,156 total). Validation fold untouched.

**4. Model training.** Two architectures were fine-tuned with transfer learning:
- *MobileNetV2:* Lightweight CNN using depthwise separable convolutions. Final classifier head replaced with a 12-class linear layer.
- *ResNet18:* 18-layer residual network. Final fully connected layer replaced with a 12-class output.

Both trained with Adam (lr=1e-4), batch size 32, for 10 epochs per fold on a GPU.

**5. Evaluation.** Per fold: accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrix. Macro F1 is the primary metric as it treats all classes equally regardless of support, making it appropriate for imbalanced data. Results reported as mean ± std across 5 folds.

In total, 6 configurations were trained and evaluated (2 architectures × 3 strategies), each for 5 folds = 30 trained models.

---

## Results *(max 500 words)*

**Summary (mean ± std across 5 folds):**

| Configuration | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| MobileNetV2 — no balance | **0.8523 ± 0.0090** | **0.8503 ± 0.0111** | 0.8519 |
| MobileNetV2 — hybrid | 0.8435 ± 0.0096 | 0.8411 ± 0.0113 | 0.8441 |
| MobileNetV2 — class weights | 0.8435 ± 0.0064 | 0.8376 ± 0.0139 | 0.8439 |
| ResNet18 — no balance | 0.8097 ± 0.0220 | 0.8046 ± 0.0250 | 0.8077 |
| ResNet18 — hybrid | 0.8113 ± 0.0240 | 0.8128 ± 0.0205 | 0.8114 |
| ResNet18 — class weights | 0.8429 ± 0.0258 | 0.8417 ± 0.0274 | 0.8415 |

MobileNetV2 consistently outperforms ResNet18. The best configuration is MobileNetV2 with no balancing (macro-F1 0.8503). Class weights substantially improved ResNet18 (0.8046 → 0.8417) but did not benefit MobileNetV2.

MobileNetV2 with no balancing wins for several reasons: the 4:1 class imbalance is mild enough that the model sees sufficient minority examples without correction; strong ImageNet pre-training generalises well from few examples (tornado: 272 samples, F1 0.93); hybrid resampling introduced noise by repeatedly augmenting the same small set of images; and class weights shifted the loss landscape inconsistently, hurting some minority classes (nuclear explosion recall dropped 0.81 → 0.75) while only modestly helping others. The no-balance model trained on the most real, diverse data. ResNet18, being weaker overall, genuinely benefits from class weighting — MobileNetV2 does not need the correction.

**Per-class highlights (best model):** Tornado (F1 0.93), wildfire (0.93), and car accident (0.92) were the easiest classes despite tornado being a minority class (272 samples) — demonstrating that visual distinctiveness matters more than data volume. Collapsed (F1 0.67) and oil spill (0.73) were the hardest, both visually ambiguous and easily confused with related categories (earthquake, flooded).

Confusion matrices (all 6 configurations) are included in the appendix. They confirm that most errors occur between structurally similar categories: collapsed↔earthquake, oil spill↔flooded.

---

## Reliability of Results *(max 300 words)*

**No held-out test set.** All evaluation is performed on cross-validation folds. This means the reported metrics are estimates of generalisation performance but are not validated on truly unseen data. A held-out test set would provide a less biased final evaluation.

**Only 10 training epochs.** Models may not have fully converged. ResNet18 in particular showed loss instability (spikes) in later epochs for the class weights and hybrid configurations, suggesting further training or a learning rate scheduler could improve and stabilise results.

**Cross-fold variance.** MobileNetV2 configurations show low variance (std ≤ 0.014), indicating reliable estimates. ResNet18 configurations have higher variance (std up to 0.027), meaning individual fold results vary more widely and the mean is a less stable estimate.

**Single random seed.** All experiments used seed 42. Results may differ under different splits; running multiple seeds would strengthen conclusions.

**Augmentation as oversampling.** In the hybrid strategy, synthetic images are created from existing ones via augmentation. These are correlated with their source images, which may reduce effective sample diversity compared to collecting new real images.

---

## Technical Depth *(max 300 words)*

This project goes beyond basic image classification in several ways:

*Class imbalance handling (DM topic):* Three strategies were systematically compared. Class-weighted loss is a technique from the data mining / learning theory literature. Hybrid resampling combines undersampling and augmentation-based oversampling, drawing on SMOTE-inspired ideas applied to image data.

*Transfer learning:* Both models use ImageNet pre-trained weights, with only the classification head retrained. This is a standard technique in deep learning not explicitly covered in the CV&IC lecture material; it was learned from the PyTorch documentation and torchvision model zoo.

*Stratified cross-validation:* Stratification ensures class proportions are preserved across folds, which is critical for imbalanced datasets. This goes beyond simple k-fold covered in the DM topic.

*Macro F1 as primary metric:* Standard accuracy is misleading on imbalanced data. Macro F1 weights all classes equally regardless of support, and is the appropriate metric for evaluating minority class performance.

---

## Conclusions & Recommendations *(max 300 words)*

**Target stakeholder:** Emergency management organisations and social media monitoring platforms seeking to automatically triage incident images.

**Main conclusions:**
1. MobileNetV2 with no balancing is the best overall configuration (macro-F1 0.8503, accuracy 0.8523), and the most stable across folds.
2. ResNet18 requires class-weighted loss to be competitive; without it, it is the worst performer with the highest variance.
3. Visual class distinctiveness predicts difficulty better than class size — tornado (minority, F1 0.93) outperforms collapsed (average, F1 0.67).

**Recommendations:**
- Deploy MobileNetV2 for production use given its superior and more stable performance.
- Invest in collecting more and more diverse images for collapsed and oil spill categories, as these are the main sources of error.
- Train for more epochs (20–30) with cosine annealing to allow full convergence before comparing strategies.

**Future research:** Explore focal loss, EfficientNet, and vision transformers; add a genuine held-out test set for unbiased evaluation.

---

## Reflection *(max 300 words)*

The main challenge was handling class imbalance rigorously without data leakage — ensuring all resampling and weight computation happened exclusively on training folds. This required careful restructuring of the training loop.

The course's CV&IC content (convolutional architectures, transfer learning, evaluation metrics) was directly applicable. The DM topic contributed knowledge on imbalanced data handling and cross-validation. Combining both perspectives was necessary and valuable.

Additional knowledge that would have been beneficial: learning rate scheduling, more advanced augmentation libraries (Albumentations), and experience with experiment tracking tools (MLflow, Weights & Biases) to better manage 30 trained models.

Claude Code (an AI coding assistant by Anthropic) was used throughout this project for: writing and iterating on the training pipeline code, generating the evaluation and visualisation cells in the notebook, writing and editing the report, and running inference to generate confusion matrix figures. All experimental design decisions, interpretation of results, and conclusions were made by the student.

---

## References *(max 1000 words)*

Weber, E., Liu, N., & Krishnamurthy, A. K. (2020). Detecting Natural Disasters, Damage, and Incidents in the Wild. *European Conference on Computer Vision (ECCV)*, Lecture Notes in Computer Science, vol 12371. Springer, Cham.

Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*.

---

## ML Issues Addressed

**Class imbalance:** The dataset has a 4:1 ratio between the largest class (car accident, 947 images) and the smallest (bicycle accident, 228). We compared three strategies: (A) no balancing, (B) class-weighted loss where per-class weights are computed as w_c = 1/count_c and applied to CrossEntropyLoss, and (C) hybrid resampling where minority classes are oversampled via augmentation and majority classes are undersampled to the median class size. MobileNetV2 with no balancing performed best — the 4:1 imbalance is mild enough that strong ImageNet pre-training compensates without correction. ResNet18 required class weighting to be competitive.

**Overfitting:** Augmentation (horizontal flip, rotation, brightness) was applied to training folds only to improve generalisation without inflating validation metrics.

---

## Generalisation Capabilities

Generalisation is evaluated using stratified 5-fold cross-validation. Stratification ensures class proportions are preserved in every fold, so results are not distorted by imbalance. Performance is reported as mean ± standard deviation across all 5 folds.

The best model (MobileNetV2, no balancing) achieves accuracy 0.8523 ± 0.0090 and macro-F1 0.8503 ± 0.0111 — the low standard deviation indicates stable generalisation across different data splits. ResNet18 configurations show higher variance (std up to 0.027), suggesting less stable generalisation.

Limitation: no held-out test set was used. All metrics are cross-validation estimates; performance on fully unseen data is not directly measured.

---

## Appendix *(max 1000 words)*

**Confusion matrices for all 6 configurations** (figures/cm_*.png):

- MobileNetV2 — no balance
- MobileNetV2 — hybrid resampling
- MobileNetV2 — class weights
- ResNet18 — no balance
- ResNet18 — hybrid resampling
- ResNet18 — class weights
