# SpamHam
NLP Spam Ham bot

## Dataset
The dataset is located at `data/SMSSpamCollection`.

## Installation
Install all required dependencies:
```bash
pip install pandas numpy scikit-learn nltk gensim sentence-transformers
```

## How to Run

**ETL Pipeline (run this first before Chadi's model):**
```bash
python3 etl.py
```

**SVM with Handcrafted Features (Samad):**
```bash
python3 SVM_HandCraftedFeatures.py
```

**Naive Bayes (Khushboo):**
```bash
python3 naive_model.py
```

**MLP with Word2Vec (Sevda):**
```bash
python3 spam_ham_MLP.py
```

**MLP with TF-IDF (Chadi):**
```bash
python3 tfidf_dense_nn.py
```

## Notes
- `SVM_HandCraftedFeatures.py` and `naive_model.py` and `spam_ham_MLP.py` can be run independently without running `etl.py` first
- `tfidf_dense_nn.py` requires `etl.py` to be run first, or the saved TF-IDF files to be present in the repo