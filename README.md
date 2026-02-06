
# BG5 - Ultrasound-Based Breast Cancer Detection Using a Segmentation-Guided Deep Learning Framework

## Team Info
- 22471A0573 — **Chandana Priya Badina** ( [LinkedIn](https://www.linkedin.com/in/chandana-priya-badina-827b97276/) )
_Work Done: Model training, evaluation, preprocessing_

- 22471A0574 — **Srilakshmi Bandi** ( [LinkedIn](https://linkedin.com/in/xxxxxxxxxx) )
_Work Done: documentation, Dataset preparation_

- 22471A05A2 — **Sri Varsha Kolisetty** ( [LinkedIn](https://www.linkedin.com/in/sri-varsha1710) )
_Work Done: Research analysis, testing_

---

## Abstract
Breast cancer remains a major health concern where early and accurate diagnosis is critical for effective treatment. This project presents an AI-based breast cancer detection system that uses ultrasound images and a segmentation-guided deep learning framework to classify tumors as benign or malignant. By combining precise tumor localization with domain-aware diagnostic guidance, the system enhances interpretability and clinical relevance. Acting as a digital decision-support tool, the proposed approach assists clinicians and medical learners in improving diagnostic accuracy, especially in resource-limited healthcare settings.

---

## Paper Reference (Inspiration)
👉 **Ultrasound Breast Image Classification Through Domain Knowledge Integration into Deep Neural Networks
– Author Names: Y. Zhang; S. Wang; P. Phillips; Z. Dong; G. Ji; J. Yang
(https://ieeexplore.ieee.org/document/10634159)**
Original conference/IEEE paper used as inspiration for the model.

---

## Our Improvement Over Existing Paper
Unlike prior ultrasound-based breast cancer studies that rely mainly on classification accuracy, our approach integrates segmentation-guided learning to explicitly localize tumor regions before diagnosis. The use of U-Net++ and SUNet improves spatial focus and interpretability. In addition, a domain-aware guidance layer provides clinically meaningful insights, making the system more practical for real-world and resource-limited healthcare settings.

---

## About the Project
What it does
Takes an ultrasound breast image and automatically identifies whether a tumor is benign or malignant using deep learning, supported by tumor segmentation.

Why it is useful
Helps in early breast cancer detection, reduces dependency on manual interpretation, improves diagnostic accuracy, and supports doctors in resource-limited healthcare settings.

Workflow: Ultrasound Image → Preprocessing → U-Net++ / SUNet Segmentation → CNN Classification Model → Diagnosis Output → Domain-Aware Clinical Guidance

---

## Dataset Used
👉 **[Breast Ultrasound Images (BUSI) Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)**

**Dataset Details:**
- 780 grayscale ultrasound images
- 3 classes: Normal, Benign, Malignant
- Pixel-wise ground truth segmentation masks for tumor regions
- Real clinical ultrasound scans with varying tumor shapes and textures
---

## Dependencies Used
- Python, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Albumentations, Matplotlib, Seaborn, scikit-learn
---

## EDA & Preprocessing
- Checked and cleaned ultrasound images for quality issues
- Studied class distribution (normal, benign, malignant)
- Resized images and masks to 128×128
- Normalized pixel values for stable training
- Applied augmentation (flip, rotation, zoom) to improve generalization
---

## Model Training Info
- **Models:** U-Net++ / SUNet (Segmentation), CNN (Classification)
- **Optimizer:** Adam
- **Loss Functions:** Dice Loss, Binary Cross-Entropy
- **Epochs:** 50
- **Learning Rate:** 1e-4
---

## Model Testing / Evaluation
- **Accuracy:** 96%
- **Precision:** 88%
- **Recall:** 84%
- **F1 Score:** 86%
- **Confusion Matrix:** Benign vs Malignant
---

## Results
- The proposed model accurately identifies tumor regions and classifies breast ultrasound images into benign and malignant categories. Compared to traditional CNN-only approaches, the segmentation-guided framework improves interpretability, robustness, and overall diagnostic reliability, making it suitable for real-world clinical use.
---

## Limitations & Future Work
- Currently validated on a single ultrasound dataset (BUSI)
- Performance may vary across different imaging devices and hospitals
- Future work includes multi-dataset validation, improved attention-based segmentation, and development of a real-time clinical decision support system
---

## Deployment Info
- The system is currently implemented as a research-level prototype for offline evaluation of breast ultrasound images. Users can input ultrasound scans to receive automated tumor segmentation and benign/malignant classification results along with clinical guidance insights. Future deployment plans include integration into a web-based clinical interface, optimization for real-time inference, and potential extension to hospital and mobile diagnostic platforms.

---
