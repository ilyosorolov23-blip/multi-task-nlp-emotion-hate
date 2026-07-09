# Multi-Task Learning with NLP: Emotion & Hate Speech Detection

This repository implements a Multi-Task Learning (MTL) NLP architecture using **Keras/TensorFlow** to simultaneously solve two distinct text classification problems with a single shared network backbone.

## 🎯 Project Objectives
1. **Emotion Detection**: Classify input text into 6 distinct emotional categories (Sadness, Joy, Love, Anger, Fear, Surprise).
2. **Hate Speech Classification**: Detect and categorize toxic text into 3 classes (Hate Speech, Offensive Speech, Neither).

---

## 📊 Dataset & Preprocessing Pipeline
To prevent the joint model from favoring one task over the other, both datasets are downsampled and balanced to exactly **12,000 samples each**:
* **Emotion Dataset (`text.csv`)**: Stratified sampling ensures exactly 2,000 balanced samples per emotion class.
* **Hate Speech Dataset (`labeled_data 2.csv`)**: Filtered down to 12,000 total rows to stabilize training.
* **Text Cleaning**: Applied advanced token sequence cleaning including alphanumeric filtering and NLTK stopword removal to improve signal-to-noise ratio.

---

## 🏗️ Model Architecture
The network leverages a shared embedding and recurrent sequence-processing backbone to learn generalized text representations, splitting into task-specific heads for individual classifications:

* **Shared Input Layers**: 
  * Tokenizer Embedding Layer
  * Bidirectional / Long Short-Term Memory (LSTM) Layer
* **Task-Specific Output Heads**:
  * **Head 1 (Emotion Branch)**: Softmax Dense Layer (6 outputs)
  * **Head 2 (Hate Speech Branch)**: Softmax Dense Layer (3 outputs)

---
