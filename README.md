# 🧘 Yoga Pose Recognition Using CNNs
### A Comparative Analysis of Custom CNN, MobileNetV2, and ResNet50 with Explainable AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![License](https://img.shields.io/badge/License-Academic-green)

---

## 📌 Project Overview

This project implements a complete comparative analysis of three CNN-based models for classifying **47 yoga pose categories** from images. It was developed as part of a Machine Learning course (Phase 2) and covers the full pipeline from data loading and preprocessing through model training, evaluation, and explainability analysis.

**Three models are compared:**
- **Custom CNN** — trained from scratch as a baseline
- **MobileNetV2** — transfer learning with two-phase fine-tuning
- **ResNet50** — transfer learning with two-phase fine-tuning

**Two Explainable AI (XAI) techniques are applied:**
- **Grad-CAM** — visualises which image regions each model focuses on
- **SHAP** — reveals pixel-level evidence used for correct and incorrect predictions

---

## 📊 Results Summary

| Model | Test Accuracy | Macro F1 | Parameters | Training Time |
|---|---|---|---|---|
| Custom CNN | 43.07% | 0.4019 | 1.33M | 749.5s |
| MobileNetV2 | 65.69% | 0.6200 | 2.60M | 614.5s |
| **ResNet50** | **73.72%** | **0.6870** | 24.12M | 720.4s |

> Random chance baseline: 1/47 = **2.1%** — all models significantly exceed this.

**Key finding:** ResNet50 achieves the highest accuracy while MobileNetV2 offers the best performance-to-cost trade-off with 9× fewer parameters and the fastest training time.

---

## 📁 Repository Structure

```
yoga-pose-recognition/
│
├── yoga_pose_cnn_final.ipynb                           # Main Kaggle notebook (complete implementation)
├── README.md                                           # This file
│
├── outputs/
│   ├── figures/                                        # All generated PDF figures
│   │   ├── figure1_dataset_samples.pdf
│   │   ├── figure3_combined_training_curves.pdf
│   │   ├── figure5_gradcam_correct.pdf
│   │   ├── figure6_gradcam_misclassified.pdf
│   │   ├── figure7_shap_custom_cnn_correct.pdf
│   │   ├── figure7_shap_custom_cnn_incorrect.pdf
│   │   ├── figure7_shap_mobilenetv2_correct.pdf
│   │   ├── figure7_shap_mobilenetv2_incorrect.pdf
│   │   ├── figure7_shap_resnet50_correct.pdf
│   │   ├── figure7_shap_resnet50_incorrect.pdf
│   │   ├── custom_cnn_training_curves.pdf
│   │   ├── custom_cnn_confusion_matrix.pdf
│   │   ├── mobilenetv2_training_curves.pdf
│   │   ├── mobilenetv2_confusion_matrix.pdf
│   │   ├── resnet50_training_curves.pdf
│   │   └── resnet50_confusion_matrix.pdf
│   │
│   └── tables/                                         # All generated CSV tables
│       ├── dataset_summary.csv
│       ├── class_image_counts.csv
│       ├── model_evaluation_results.csv
│       ├── final_comparative_table.csv
│       ├── top_confused_pairs.csv
│       ├── misclassification_examples.csv
│       ├── research_question_answers.csv
│       ├── explainability_ratings.csv
│       ├── Custom CNN_classification_report.csv
│       ├── MobileNetV2_classification_report.csv
│       ├── ResNet50_classification_report.csv
│       ├── custom_cnn_training_history.csv
│       ├── mobilenetv2_training_history.csv
│       └── resnet50_training_history.csv
```

---

## 📦 Dataset

**Name:** Yoga Posture Dataset  
**Source:** Kaggle — [https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)  
**Author:** tr1gg3rtrash  

| Property | Value |
|---|---|
| Number of classes | 47 yoga pose categories |
| Total images | ~2,756 |
| Average images per class | ~58.6 |
| Image type | RGB (.jpg, .jpeg, .png) |
| Input size used | 224 × 224 pixels |
| Task type | Multi-class image classification |

**Dataset split used:**
- Training: 70% (~1,929 images)
- Validation: 15% (~413 images)
- Test: 15% (~414 images)

---

## 🔬 Research Questions

| RQ | Question |
|---|---|
| RQ1 | How accurately does a custom CNN classify yoga poses? |
| RQ2 | Does transfer learning improve classification over a custom CNN? |
| RQ3 | Which model provides the best performance vs. computational cost trade-off? |
| RQ4 | Which image regions does each model focus on for correct predictions (Grad-CAM)? |
| RQ5 | Do Grad-CAM patterns differ between the custom CNN and transfer learning models? |
| RQ6 | What do SHAP explanations reveal about visual evidence used by each model? |
| RQ7 | Can explainability methods diagnose why certain poses are misclassified? |

---

## 🛠️ Implementation Details

### Models

**Custom CNN**
- 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- Each block: Conv2D × 2, BatchNorm, MaxPooling, Dropout
- Head: GlobalAveragePooling → Dense(512) → Dropout(0.4) → Dense(47, softmax)
- Total parameters: 1,331,791

**MobileNetV2 (Transfer Learning)**
- Base: MobileNetV2 pretrained on ImageNet
- Phase 1: Base frozen, head trained for 15 epochs at lr=1e-3
- Phase 2: Last 30 layers unfrozen, fine-tuned for 25 epochs at lr=1e-5
- Total parameters: 2,597,999

**ResNet50 (Transfer Learning)**
- Base: ResNet50 pretrained on ImageNet
- Phase 1: Base frozen, head trained for 15 epochs at lr=1e-3
- Phase 2: Last 40 layers unfrozen, fine-tuned for 25 epochs at lr=1e-5
- Total parameters: 24,124,335

### Training Configuration

| Setting | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | Categorical Cross-Entropy |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| Class weighting | sklearn compute_class_weight (balanced) |
| Random seed | 42 |

### Data Augmentation (Training only)
- Random horizontal flip
- Random rotation ±10%
- Random zoom 10%
- Random translation 10%

---

## 🚀 How to Run

### Option 1 — Run on Kaggle (Recommended)

1. Go to [Kaggle](https://www.kaggle.com) and sign in
2. Create a new notebook and import `yoga_pose_cnn_final.ipynb`
3. Add the dataset: search for `yoga-posture-dataset` by tr1gg3rtrash and click Add
4. In Cell 3, update `USER_DATASET_PATH` to match the path from the dataset panel
5. Enable GPU (T4) and Internet in Session options
6. Click **Run All**

### Option 2 — Run Locally

**Requirements:**
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python shap
```

**Steps:**
1. Download the dataset from Kaggle
2. Update `USER_DATASET_PATH` in Cell 3 to your local dataset path
3. Update `OUTPUT_DIR` in Cell 3 to your preferred output directory
4. Run all cells in order

---

## 📋 Dependencies

| Package | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| NumPy | Numerical operations |
| Pandas | Data management and CSV export |
| Matplotlib | Figure generation |
| scikit-learn | Metrics, class weights, confusion matrix |
| OpenCV (cv2) | Grad-CAM heatmap overlay |
| SHAP | Pixel-level explainability |

---

## 📈 Output Files

The notebook generates **44 output files** automatically:

- **19 PDF figures** — training curves, confusion matrices, Grad-CAM visualizations, SHAP explanations
- **22 CSV tables** — evaluation metrics, classification reports, training histories, research question answers
- **3 model files** — saved Keras models (.keras format)

All outputs are saved to `/kaggle/working/outputs/` when run on Kaggle.

---

## 🔍 Explainability Analysis

### Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which spatial regions of an image the model focuses on when making a prediction. The last convolutional layer is used for each model:

| Model | Last Conv Layer |
|---|---|
| Custom CNN | conv2d_15 |
| MobileNetV2 | Conv_1 |
| ResNet50 | conv5_block3_3_conv |

### SHAP
SHAP (SHapley Additive exPlanations) identifies which individual pixels contributed positively or negatively to the model's prediction. Applied to both correctly classified and misclassified images using GradientExplainer with 30 background samples.

### Notable Misclassifications
The analysis revealed that errors occur primarily between geometrically similar poses:

| True Pose | Predicted As | Reason |
|---|---|---|
| Trikonasana | Virabhadrasana Two | Both wide-legged stances with extended arms |
| Virabhadrasana One | Anjaneyasana | Both deep lunge poses with raised arms |
| Paschimottanasana | Ardha Matsyendrasana | Both seated forward/twist poses |

---

## 📚 References

- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Yoga Posture Dataset — https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset

---

## 📝 Course Information

**Course:** Machine Learning  
**Phase:** 2 — Proposal, Code Development, and Technical Implementation  
**Framework:** TensorFlow / Keras  
**Platform:** Kaggle Notebook (GPU T4)  

---

*For questions or issues with running the code, please open a GitHub issue.*
