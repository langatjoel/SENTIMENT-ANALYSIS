# 📊 Sentiment Analysis on Product Reviews (Logistic Regression Model)

## Project Overview

This project successfully implemented a **Sentiment Analysis** pipeline to classify customer product reviews into three discrete sentiment categories: **Positive**, **Negative**, and **Neutral**. The work adheres to a structured, accelerated data science workflow.

* **Goal:** Multi-class text classification (3 classes).
* **Dataset:** Amazon/IMDB Reviews Dataset (`amazon_reviews.csv`).
* **Final Model:** **Logistic Regression**, selected for its efficiency and strong baseline performance on sparse, high-dimensional text data.

---

## 🛠 Methodology & Pipeline

The entire analysis is detailed in the accompanying Jupyter Notebook/Python script.

### 1. Data Preparation and Cleaning (Weeks 1 & 2)

* **Data Quality:** Critical rows missing `review_text` or the target `sentiment` were removed.
* **EDA Insight:** Initial analysis revealed a significant **class imbalance**, with the Neutral class being the minority. This finding was crucial for guiding the model selection.
* **Text Cleaning:** Reviews were normalized (lowercased) and cleaned of punctuation/symbols to prepare the text for vectorization.

### 2. Feature Engineering: TF-IDF (Week 3)

* The cleaned text was transformed into a numerical matrix using the **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorizer. This method assigns weight to words based on their relevance across the entire dataset, creating meaningful features for the model.
* **Data Split:** The vectorized features were split into $80\%$ training and $20\%$ testing sets using **stratification** to preserve the original class distribution in both partitions.

### 3. Model Building (Week 4)

* **Model Choice:** Logistic Regression was chosen for its high interpretability and efficiency on sparse TF-IDF data.
* **Tuning for Imbalance:** The model was trained with the hyperparameter `class_weight='balanced'`. This adjustment is crucial as it automatically weights the minority classes (like Neutral) more heavily, preventing the model from simply ignoring them during training.

---

## 📈 Results & Key Findings (Week 5)

The final model was evaluated on the test set, with the **Weighted F1-Score** used as the overall performance metric.

### Key Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | [0.5138]% |


### Key Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | **0.5138** |
| **Weighted F1-Score** | **0.5186** | 

| Sentiment Class | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Negative** | 0.59 | 0.49 | 0.54 |
| **Neutral** | 0.39 | 0.51 | 0.44 |
| **Positive** | 0.52 | 0.54 | 0.53 |

### Visual Evaluation

The Confusion Matrix provides a detailed visual breakdown of the model's classification performance.



[Image of Confusion Matrix for Logistic Regression]


### Conclusion

* **Effective Use of Class Weighting:** The strategy of using `class_weight='balanced'` was successful. It significantly improved the **Recall** for the minority Neutral class, ensuring the model could detect subtle or ambiguous reviews, even if its precision for that class remained lower.
* **Robust Negative Detection:** The model remains most reliable and confident when predicting Negative sentiment, achieving the highest F1-Score for this class.
* **Future Work:** This Logistic Regression model serves as an excellent, lightweight baseline. For substantial performance improvements, future iterations should explore Deep Learning models (e.g., Bi-LSTMs or BERT) to capture the deep contextual relationships in the review text.

---

## 👥 Group Project Contribution

This project was a collaborative effort by the following 8 members. The full task breakdown is documented in `project_work_breakdown.md`.

| Member | Key Contribution Areas |
| :--- | :--- |
| **Natasha Gicheha** | Problem Statement, Logistic Regression Training, Accuracy Reporting. |
| **Mercy Muthoni** | Data Sourcing, Data Cleaning (Duplicates), Confusion Matrix Generation. |
| **Mary Edna** | Data Inspection, Class Distribution Analysis, F1-Score Reporting. |
| **Sydney Ndolo** | Data Loading, Visualization Generation. |
| **Isaac Nthusi** | Directory Structure Creation, Final Script Compilation. |
| **Joel Kipruto** | GitHub Setup, **TF-IDF Vectorization**, Final Report Summary. |
| **Henry Modi** | Critical Data Cleaning, Train-Test Split, Hyperparameter Tuning (Class Weighting). |
| **Karina Murage** | Feature Documentation, Model Finalization, GitHub Submission. |

---

## 🚀 Setup and Usage

To run this project locally, ensure you have Python 3.8+ installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt