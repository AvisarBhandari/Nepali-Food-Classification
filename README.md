# 🍛 Nepali Food Classifier

A deep learning image classification project to recognize Nepali food dishes using **transfer learning (ResNet18)**.
This project compares two training strategies based on different data augmentation techniques.

---

### Streamlit App

Run the web app: [View Project](https://avisarbhandari-nepali-food-classific-srcstreamlit-appapp-fdmlas.streamlit.app/)

---

### FastAPI 

```
POST /predict
```

<img width="1413" height="421" alt="pasted file" src="https://github.com/user-attachments/assets/9b9747b0-78f9-40e2-af54-5593c2642904" />

---

## Project Overview

This project trains two separate models using the same architecture **(ResNet18)** but different data preprocessing strategies:

|Model|	Description|
|---|---|
|Model 0| (Basic Augmentation)	Minimal augmentation for baseline performance|
|Model 1 |(Strong Augmentation)	Advanced augmentation for better generalization|

The goal is to analyze how augmentation impacts:

* Accuracy
* Overfitting
* Generalization
* Robustness

---

## Project Structure

```txt
project/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── predict.py
|   ├── train.py
│   └── streamlit_app
│    
├── models/
└── README.md
```
---

## Dataset
Source: Kaggle Food Dataset (custom Nepali food subset) [Data Source](https://www.kaggle.com/datasets/rajeevpaudel1/nepali-food-image-dataset)
### Classes used:

* momo
* dal_bhat
* sel_roti
* chowmein
* thukpa
* chatamari
* yomari

mage format: RGB \
Input size: 224 x 224

---

## Model

### Base architecture:

**ResNet18** 

Transfer learning used with frozen feature extractor and custom classifier head.

---

### Data Augmentation Strategies
 Model 0 — Basic Augmentation

Used for baseline training:
```
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

Characteristics:
* No heavy augmentation
* Preserves original data distribution
* Faster convergence
* Higher risk of overfitting

  Model B — Strong Augmentation

Used for better generalization:

```
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])
```

Characteristics:
*More robust to real-world variation
*Reduces overfitting
*Slightly slower convergence
*Better generalization

---

### Training Details
Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 1e-3 (fine-tuning)
Epochs: 15
Batch Size: 32

---

## Training Results

### Accuracy Plot

![alt text](https://github.com/AvisarBhandari/Nepali-Food-Classification/blob/2e78081c67f514886fcce4aca8451a362deafd64/model/Accuracy.png)

### Loss Plot

![alt text](https://github.com/AvisarBhandari/Nepali-Food-Classification/blob/2e78081c67f514886fcce4aca8451a362deafd64/model/Loss.png)

---

### Confusion Matrix

Model 0 

![alt text](https://github.com/AvisarBhandari/Nepali-Food-Classification/blob/2e78081c67f514886fcce4aca8451a362deafd64/model/model_0.png)

Model 1

![alt text](https://github.com/AvisarBhandari/Nepali-Food-Classification/blob/2e78081c67f514886fcce4aca8451a362deafd64/model/model_1.png)

---

### Inference Pipeline

Steps:

1. Load image
2. Resize to 224x224
3. Normalize
4. Forward pass through ResNet18
5. Apply Softmax
6. Get prediction (Argmax)

---







