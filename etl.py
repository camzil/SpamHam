# Run once — installs all required libraries
import os
import scipy.sparse as sp
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import string
import re

# Download required NLTK data (only needed once — safe to re-run)
for pkg in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    nltk.download(pkg, quiet=True)

print("All imports successful.")


def load_data(filepath: str = "SMSSpamCollection.txt") -> pd.DataFrame:
    """
    Load the SMS Spam Collection dataset.

    Parameters
    ----------
    filepath : str
        Path to the tab-separated dataset file.

    Returns
    -------
    pd.DataFrame with columns: label (str), message (str)
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",   # dataset uses latin-1 encoding
    )
    print(f"Total messages loaded : {len(df):,}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nSpam rate: {df['label'].eq('spam').mean():.1%}")
    return df


# ── Run it ────────────────────────────────────────────────────────────────────
df = load_data("data/SMSSpamCollection")
df.head(5)


STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = True,
) -> str:
    """
    Clean and normalise a single SMS message.

    Parameters
    ----------
    text             : Raw SMS string
    remove_stopwords : Remove common English words (the, is, at, ...)
    use_stemming     : Apply Porter stemmer (aggressive, fast)
    use_lemmatization: Apply WordNet lemmatizer (gentler, real words)

    Returns
    -------
    Cleaned string ready for vectorisation.
    """
    # 1. Lowercase — treats "FREE" and "free" as the same word
    text = text.lower()

    # 2. Remove URLs — they add noise, not signal
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove phone numbers (common spam pattern, but not the word itself)
    text = re.sub(r"\b\d[\d\s\-]{6,}\d\b", "", text)

    # 4. Remove punctuation and digits — keep letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenise into individual words
    tokens = word_tokenize(text)

    # 6. Remove stopwords — high-frequency words that carry little meaning
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # 7. Stem or Lemmatize — reduce inflected forms to a base form
    if use_stemming:
        tokens = [stemmer.stem(t) for t in tokens]
    elif use_lemmatization:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame):
    """
    Apply preprocessing to every row and encode labels.

    Label encoding:  ham → 0,  spam → 1
    """
    df = df.copy()
    df["clean_message"] = df["message"].apply(preprocess_text)

    le = LabelEncoder()
    le.fit(["ham", "spam"])              # explicit ordering: ham=0, spam=1
    df["label_encoded"] = le.transform(df["label"])

    return df, le


# ── Run it ────────────────────────────────────────────────────────────────────
df, label_encoder = preprocess_dataframe(df)

# Show before / after comparison
comparison = df[["label", "message", "clean_message"]].head(6)
for _, row in comparison.iterrows():
    print(f"[{row.label.upper():4s}]  RAW: {row.message[:60]}")
    print(f"        CLEAN: {row.clean_message[:60]}")


def split_data(df: pd.DataFrame):
    """
    Split into Train (60%) / Dev (20%) / Test (20%) with stratification.

    The split is done BEFORE any feature extraction to prevent data leakage.

    Returns
    -------
    X_train, X_dev, X_test : numpy arrays of cleaned message strings
    y_train, y_dev, y_test : numpy arrays of integer labels (0=ham, 1=spam)
    """
    X = df["clean_message"].values
    y = df["label_encoded"].values

    # Step 1: carve out 20% for the final test set
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Step 2: split remaining 80% into 75% train / 25% dev → 60/20 overall
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_traindev, y_traindev,
        test_size=0.25, random_state=42, stratify=y_traindev
    )

    n = len(X)
    print("Split sizes (stratified):")
    print(
        f"  Train : {len(X_train):,}  ({len(X_train)/n:.0%})  spam: {y_train.mean():.1%}")
    print(
        f"  Dev   : {len(X_dev):,}   ({len(X_dev)/n:.0%})  spam: {y_dev.mean():.1%}")
    print(
        f"  Test  : {len(X_test):,}   ({len(X_test)/n:.0%})  spam: {y_test.mean():.1%}")

    return X_train, X_dev, X_test, y_train, y_dev, y_test


# ── Run it ────────────────────────────────────────────────────────────────────
X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)


def get_bow_features(X_train, X_dev, X_test, max_features: int = 5000):
    """
    Build Bag-of-Words feature matrices using CountVectorizer.

    Each message → sparse vector of word/bigram counts.
    Vectoriser is fit ONLY on training data to prevent leakage.

    Parameters
    ----------
    X_train, X_dev, X_test : arrays of cleaned message strings
    max_features           : vocabulary size cap

    Returns
    -------
    Sparse scipy matrices + fitted vectoriser
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),    # include both single words and 2-word phrases
        analyzer="word",
        strip_accents="unicode",
    )

    # IMPORTANT: fit_transform on train, transform-only on dev/test
    X_train_bow = vectorizer.fit_transform(X_train)
    X_dev_bow = vectorizer.transform(X_dev)
    X_test_bow = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()

    print(f"BoW matrix shape     (train) : {X_train_bow.shape}")
    print(f"Vocabulary size              : {len(feature_names):,}")
    print(
        f"Matrix density               : {X_train_bow.nnz / (X_train_bow.shape[0]*X_train_bow.shape[1]):.4%}")
    print(f"\nFirst 10 vocabulary terms    : {feature_names[:10].tolist()}")
    print(f"Last  10 vocabulary terms    : {feature_names[-10:].tolist()}")

    return X_train_bow, X_dev_bow, X_test_bow, vectorizer


# ── Run it ────────────────────────────────────────────────────────────────────
X_tr_bow, X_dv_bow, X_te_bow, bow_vec = get_bow_features(
    X_train, X_dev, X_test)

# Peek at a single message as a vector
sample_idx = 2
print(f"\nSample message   : '{X_train[sample_idx]}'")
row = X_tr_bow[sample_idx]
nonzero_features = [(bow_vec.get_feature_names_out()[i], row[0, i])
                    for i in row.nonzero()[1]]
print(
    f"Non-zero entries : {sorted(nonzero_features, key=lambda x: -x[1])[:10]}")


def get_tfidf_features(X_train, X_dev, X_test, max_features: int = 5000):
    """
    Build TF-IDF feature matrices.

    Downweights very common words and rewards rare informative ones.
    Uses smooth IDF and sublinear TF dampening.

    Returns
    -------
    Sparse scipy matrices + fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # log(1 + tf) instead of raw tf
        analyzer="word",
        strip_accents="unicode",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    print(f"TF-IDF matrix shape  (train) : {X_train_tfidf.shape}")
    print(
        f"Matrix density               : {X_train_tfidf.nnz / (X_train_tfidf.shape[0]*X_train_tfidf.shape[1]):.4%}")

    # Show the 10 highest-IDF words (rarest, most distinctive)
    top_idf_idx = np.argsort(idf_scores)[-10:][::-1]
    print("\nTop 10 highest-IDF terms (most distinctive):")
    for idx in top_idf_idx:
        print(f"  {feature_names[idx]:25s}  idf={idf_scores[idx]:.3f}")

    # Show the 10 lowest-IDF words (most common, downweighted)
    bot_idf_idx = np.argsort(idf_scores)[:10]
    print("\nBottom 10 lowest-IDF terms (most common, downweighted):")
    for idx in bot_idf_idx:
        print(f"  {feature_names[idx]:25s}  idf={idf_scores[idx]:.3f}")

    return X_train_tfidf, X_dev_tfidf, X_test_tfidf, vectorizer


# ── Run it ────────────────────────────────────────────────────────────────────
X_tr_tfidf, X_dv_tfidf, X_te_tfidf, tfidf_vec = get_tfidf_features(
    X_train, X_dev, X_test)


def get_word2vec_features(X_train, X_dev, X_test, embedding_dim: int = 100):
    """
    Train Word2Vec on the training corpus, then average word vectors per message.

    Produces dense numpy arrays of shape (n_samples, embedding_dim).

    Word2Vec is trained ONLY on X_train — dev/test words not in the training
    vocabulary get zero vectors (out-of-vocabulary handling).

    Parameters
    ----------
    embedding_dim : int
        Size of each word vector (100 is a good default for short texts).

    Returns
    -------
    Dense numpy arrays (n_samples, 100) + trained Word2Vec model
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("gensim not installed. Run: pip install gensim")
        return None, None, None, None

    # Word2Vec expects a list of token lists, not strings
    tokenized_train = [text.split() for text in X_train]

    print("Training Word2Vec on training corpus...")
    w2v_model = Word2Vec(
        sentences=tokenized_train,
        vector_size=embedding_dim,
        window=5,               # context window size
        min_count=2,            # ignore very rare words
        workers=4,              # parallel training
        sg=1,                   # 1=skip-gram, 0=CBOW
        epochs=15,
        seed=42,
    )
    print(f"Vocabulary size (Word2Vec) : {len(w2v_model.wv):,} words")

    def average_embedding(tokens):
        """Average vectors for all known words; return zeros for OOV messages."""
        vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        return np.zeros(embedding_dim)   # out-of-vocabulary fallback

    def embed_corpus(texts):
        return np.array([average_embedding(t.split()) for t in texts])

    X_train_w2v = embed_corpus(X_train)
    X_dev_w2v = embed_corpus(X_dev)
    X_test_w2v = embed_corpus(X_test)

    print(f"Word2Vec feature shape (train) : {X_train_w2v.shape}")
    print(f"                      (dev)   : {X_dev_w2v.shape}")

    # Show most similar words to 'free' and 'win' — classic spam triggers
    for anchor in ["free", "win", "call"]:
        if anchor in w2v_model.wv:
            similar = w2v_model.wv.most_similar(anchor, topn=5)
            print(f"\nWords most similar to '{anchor}':")
            for word, score in similar:
                print(f"  {word:20s} {score:.3f}")

    return X_train_w2v, X_dev_w2v, X_test_w2v, w2v_model


# ── Run it ────────────────────────────────────────────────────────────────────
X_tr_w2v, X_dv_w2v, X_te_w2v, w2v_model = get_word2vec_features(
    X_train, X_dev, X_test)


def get_bert_features(X_train, X_dev, X_test,
                      model_name: str = "all-MiniLM-L6-v2"):
    """
    Encode messages with a pre-trained Sentence-BERT model.

    Each message → dense vector of 384 dimensions.
    No training required — the model is already pre-trained.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier. 'all-MiniLM-L6-v2' is fast and accurate.

    Returns
    -------
    Dense numpy arrays (n_samples, 384) + loaded model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "sentence-transformers not installed.  Run: pip install sentence-transformers")
        return None, None, None, None

    print(f"Loading BERT model '{model_name}'...")
    bert = SentenceTransformer(model_name)

    print("Encoding training messages...")
    X_train_bert = bert.encode(
        list(X_train), show_progress_bar=True, batch_size=64, convert_to_numpy=True
    )
    print("Encoding dev messages...")
    X_dev_bert = bert.encode(
        list(X_dev),   show_progress_bar=True, batch_size=64, convert_to_numpy=True
    )
    print("Encoding test messages...")
    X_test_bert = bert.encode(
        list(X_test),  show_progress_bar=True, batch_size=64, convert_to_numpy=True
    )

    print(f"\nBERT feature shape (train) : {X_train_bert.shape}")
    print(f"               (dev)     : {X_dev_bert.shape}")

    # Show cosine similarity between a ham and a spam example
    from numpy.linalg import norm
    def cosine_sim(a, b): return np.dot(a, b) / (norm(a) * norm(b))
    spam_idx = np.where(y_train == 1)[0][0]
    ham_idx = np.where(y_train == 0)[0][0]
    sim = cosine_sim(X_train_bert[spam_idx], X_train_bert[ham_idx])
    print(f"\nCosine similarity (spam vs ham example) : {sim:.4f}")
    print("  (lower = more distinct — BERT distinguishes spam/ham semantically)")

    return X_train_bert, X_dev_bert, X_test_bert, bert


# ── Run it (requires sentence-transformers) ───────────────────────────────────
X_tr_bert, X_dv_bert, X_te_bert, bert_model = get_bert_features(
    X_train, X_dev, X_test)


def evaluate_features(X_tr, X_dv, y_tr, y_dv, name: str):
    """
    Fit a Logistic Regression and print a classification report on the dev set.

    Used purely as a sanity check — not the final model.

    Parameters
    ----------
    X_tr, X_dv : feature matrices (sparse or dense)
    y_tr, y_dv : integer label arrays
    name       : label for printing

    Returns
    -------
    Fitted LogisticRegression classifier
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_dv)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_dv, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_dv, y_pred)
    print(f"Confusion matrix:")
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}  (ham predicted as spam)")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}  (spam caught correctly)")

    return clf


# ── Run sanity checks ─────────────────────────────────────────────────────────
print("Running sanity checks on Dev set...")
_ = evaluate_features(X_tr_bow,   X_dv_bow,   y_train,
                      y_dev, "Bag-of-Words (BoW)")
_ = evaluate_features(X_tr_tfidf, X_dv_tfidf, y_train, y_dev, "TF-IDF")

if X_tr_w2v is not None:
    _ = evaluate_features(X_tr_w2v, X_dv_w2v, y_train,
                          y_dev, "Word2Vec avg embedding")

if X_tr_bert is not None:
    _ = evaluate_features(X_tr_bert, X_dv_bert, y_train,
                          y_dev, "BERT (Sentence-Transformer)")


print("Saving feature matrices...")

# ── Sparse matrices (BoW and TF-IDF) ─────────────────────────────────────────
sp.save_npz("X_train_bow.npz",   X_tr_bow)
sp.save_npz("X_dev_bow.npz",     X_dv_bow)
sp.save_npz("X_test_bow.npz",    X_te_bow)
print("Saved BoW matrices.")

sp.save_npz("X_train_tfidf.npz", X_tr_tfidf)
sp.save_npz("X_dev_tfidf.npz",   X_dv_tfidf)
sp.save_npz("X_test_tfidf.npz",  X_te_tfidf)
print("Saved TF-IDF matrices.")

# ── Dense arrays (Word2Vec, BERT) ─────────────────────────────────────────────
if X_tr_w2v is not None:
    np.save("X_train_w2v.npy", X_tr_w2v)
    np.save("X_dev_w2v.npy",   X_dv_w2v)
    np.save("X_test_w2v.npy",  X_te_w2v)
    print("Saved Word2Vec arrays.")

if X_tr_bert is not None:
    np.save("X_train_bert.npy", X_tr_bert)
    np.save("X_dev_bert.npy",   X_dv_bert)
    np.save("X_test_bert.npy",  X_te_bert)
    print("Saved BERT arrays.")

# ── Labels ────────────────────────────────────────────────────────────────────
np.save("y_train.npy", y_train)
np.save("y_dev.npy",   y_dev)
np.save("y_test.npy",  y_test)
print("Saved label arrays.")

print("\n" + "="*50)
print("  All files saved. Summary:")
print("="*50)
for fname in sorted(os.listdir(".")):
    if fname.endswith((".npz", ".npy")):
        size_kb = os.path.getsize(fname) / 1024
        print(f"  {fname:35s}  {size_kb:7.1f} KB")

print("\nQuick reload verification:")
X_check = sp.load_npz("X_train_tfidf.npz")
y_check = np.load("y_train.npy")
print(f"  X_train_tfidf shape : {X_check.shape}  ✓")
print(f"  y_train shape       : {y_check.shape}  ✓")
