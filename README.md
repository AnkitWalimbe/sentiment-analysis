# Sentiment Analysis on Amazon Fashion Reviews 

This project applies Natural Language Processing (NLP) to analyze customer sentiment in **Amazon Fashion reviews**.  
It combines **VADER sentiment analysis** with a basic **machine learning model (SGDClassifier + TF-IDF)** to classify reviews into **positive, neutral, or negative** categories.

---

##  Project Overview
- **Dataset**: Amazon Fashion Reviews (JSONL format, >1M reviews).  
  The dataset is too large to host here – see [data/README.md](data/README.md) for download instructions.  
- **Goal**: Explore review text, extract sentiment, and compare rule-based (VADER) vs machine learning approaches.  
- **Why it matters**: Customer sentiment is a key driver for **returns reduction**, **CX insights**, and **product feedback loops**.

---

##  Project Highlights
- Processed **~23K Amazon Fashion reviews**.  
- Applied **VADER sentiment analysis** to classify reviews.  
- Built a baseline **ML model** using TF-IDF + SGDClassifier.  
- Visualized sentiment distributions and agreement between VADER and ML.  
- Saved key plots and reports for reproducibility.  

---

##  Repository Structure
```plaintext
sentiment-analysis/
│── data/               <- raw data (ignored), dataset instructions in README.md
│── notebooks/          <- Jupyter notebooks
│── outputs/            <- figures and saved results
│── reports/            <- generated CSVs (ignored in Git)
│── .gitignore
│── README.md           <- project documentation

```

---

##  Workflow
1. **Data Preprocessing**  
   - Converted timestamps into structured `date` and `time`.  
   - Selected relevant fields: `rating`, `title`, `text`, `asin`, `user_id`.  
   - Handled missing values.  

2. **Exploratory Data Analysis**  
   - Visualized rating distributions.  
   - Checked review counts by star ratings.  

3. **Sentiment Analysis with VADER**  
   - Applied **NLTK VADER** to calculate `pos`, `neg`, `neu`, and `compound` scores.  
   - Classified reviews into **positive, negative, neutral** buckets.  

4. **Machine Learning Model (Baseline)**  
   - Used **TF-IDF Vectorizer** to transform review text.  
   - Trained **SGDClassifier** (linear SVM) on balanced subsets.  
   - Evaluated with accuracy, classification report, and confusion matrix.  

5. **Comparison: VADER vs ML**  
   - Created a comparison dataset of predictions.  
   - Observed significant **bias toward positive reviews** due to data imbalance.  

---

##  Key Outputs
- **Figures**:  
  - Rating distribution plots.  
  - Sentiment distribution (VADER).  
  - Confusion matrices (ML model & VADER vs ML).  
- **Reports**:  
  - Sentiment comparison CSV.  

All outputs are saved in the `outputs/figures/` and `reports/` directories.  

---

##  Learnings
- VADER works well as a quick, rule-based baseline for sentiment.  
- ML models trained on imbalanced data struggle to classify negative/neutral reviews.  
- TF-IDF is limited in capturing context — embeddings like Word2Vec, BERT, or GloVe could improve results.  
- Undersampling balances training data but can **lose information**. Alternative approaches (e.g., SMOTE) may perform better.  

---

##  Next Steps
- Experiment with **Word2Vec / BERT embeddings**.  
- Test **Random Forests / Neural Networks** for better classification.  
- Fine-tune sentiment thresholds in VADER.  
- Automate preprocessing and evaluation via a pipeline.  

---

##  Tech Stack
- **Python** (pandas, NumPy, matplotlib, seaborn, tqdm)  
- **NLP**: NLTK, VADER Sentiment  
- **ML**: scikit-learn, imbalanced-learn  
- **Visualization**: seaborn, matplotlib  

---

##  Citation
Dataset reference:  

Hou, Yupeng et al. (2024).  
*Bridging Language and Items for Retrieval and Recommendation.*  
[arXiv:2403.03952](https://arxiv.org/abs/2403.03952)  

---
