# Fake News Detection using Support Vector Machines (SVM)

## Overview
Fake news spreads **6x faster** than real news, shaping opinions and fueling misinformation. This project **fights back with AI**, using a **Support Vector Machine (SVM)** to classify news articles as **real or fake**. With **multilingual support**, it processes both **English and Chinese news**, leveraging **batch translation, text preprocessing, and machine learning** to deliver accurate predictions.

## Features
✅ **Multilingual Support** – Detects fake news in **both English & Chinese**  
✅ **Optimized Batch Translation** – Faster Weibo dataset processing  
✅ **SVM with TF-IDF** – Powerful text classification with efficient feature extraction  
✅ **Real-time Prediction** – Input any news article and get an instant **real/fake** verdict  

## Datasets Used
| Dataset | Source | Samples | Columns |
|---------|--------|---------|----------|
| **Weibo21** | Chinese social media (Weibo) | ~20,000 | `content`, `label` (0 = Real, 1 = Fake) |
| **Kaggle Fake News** | English news articles | ~44,000 | `title`, `text`, `label` (0 = Real, 1 = Fake) |

Weibo’s **Chinese text is batch-translated** before being merged with Kaggle’s English dataset.

## Installation
Run the following to install dependencies:
```bash
pip install deep_translator nltk scikit-learn pandas numpy
```

## Steps in the Notebook
### **1️⃣ Install & Import Libraries**
- Loads necessary Python libraries (NLTK, Scikit-learn, Pandas, Deep Translator).

### **2️⃣ Load & Merge Datasets**
- Loads **Weibo21 & Kaggle Fake News** datasets.
- **Batch translates Chinese text** to English for efficiency.

### **3️⃣ Preprocess Text Data**
- **Cleans text** (lowercasing, punctuation & stopword removal, lemmatization).

### **4️⃣ Convert Text to TF-IDF Vectors**
- Converts raw text into **numerical representation** for SVM.

### **5️⃣ Train the SVM Model**
- Uses **TF-IDF features** and trains an **SVM classifier**.
- Evaluates performance using **accuracy, precision, recall, and F1-score**.

### **6️⃣ Test with a Sample News Input**
- Input a **custom news article** and check if it’s **real or fake**.

## Running the Model
To test a news article:
```python
sample_text = "Breaking: Scientists discover a new planet with signs of life."
print(f'Prediction: {predict_news(sample_text)}')
```

## Performance Metrics
✅ **Precision** – High precision means fewer real news misclassified as fake.  
✅ **Recall** – Ensures most fake news is detected.  
✅ **F1-Score** – Balances precision & recall for a robust model.  

## Future Improvements
🚀 **Fine-tune hyperparameters (C, kernel) for better performance**  
🚀 **Upgrade to deep learning (BERT, LSTM) for contextual analysis**  
🚀 **Deploy as a real-time web app with API integration**  

## License
This project is licensed under the **MIT License**. Use, modify, and improve freely!  

---
### **Why This Matters**
This isn’t just about an algorithm—it’s about **fighting misinformation**. When fake news spreads unchecked, trust in media crumbles. But with **AI-powered truth detection**, we can take back control. Because in a world full of noise, **facts still matter.** 📰✨


