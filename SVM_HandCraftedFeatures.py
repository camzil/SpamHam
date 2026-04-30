import re
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# load the dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                 names=["label", "message"], encoding="latin-1")
# remove duplicates and create binary label
df = df.drop_duplicates().reset_index(drop=True)
# create binary label: 1 for spam, 0 for ham
df["label_int"] = (df["label"] == "spam").astype(int)




# x_all is the raw text messages which will be transformed into features
X_all = df["message"].values
# y_all is the binary label (1 for spam, 0 for ham)
y_all = df["label_int"].values



# split into train/dev/test (60/20/20) with stratification to maintain class balance
X_traindev, X_test, y_traindev, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

# further split train/dev into 75/25 to get final train/dev sets (which results in 60/20/20 overall)
X_train, X_dev, y_train, y_dev = train_test_split(
    X_traindev, y_traindev, test_size=0.25, random_state=42, stratify=y_traindev)




# common spam indicators to look for in the text
SPAM_KEYWORDS = [
    "free", "win", "winner", "prize", "won", "claim", "urgent",
    "cash", "offer", "guaranteed", "selected", "reward",
]





# extract hand-crafted features from the text
def extract_features(text):
    t = text.lower()
    # check if there are any urls in the text
    has_url = int(bool(re.search(r"http\S+|www\.\S+", text, re.IGNORECASE)))
    
    # check for currency symbols. not just us
    has_currency = int(bool(re.search(r"[£$€¥]", text)))
    
    # check for phone numbers or long sequences of digits
    has_phone = int(bool(re.search(r"\b\d[\d\s\-]{5,}\d\b", text)))
    
    # check for 4-6 digit shortcodes
    has_shortcode = int(bool(re.search(r"\b\d{4,6}\b", text)))
    
    # check for words that show opting out
    has_optout = int(bool(re.search(r"\bstop\b|\bcancel\b|\bunsubscribe\b", t)))
    
    # count how many spam keywords appear in the text
    count = 0
    for keyword in SPAM_KEYWORDS:
        if keyword in t:
            count += 1
    
    return [has_url, has_currency, has_phone, has_shortcode,
            has_optout, count]




# extract features for all splits
X_train_f = np.array([extract_features(t) for t in X_train])
X_dev_f   = np.array([extract_features(t) for t in X_dev])
X_test_f  = np.array([extract_features(t) for t in X_test])




# train an svm classifier with rbf kernel
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, probability=True, random_state=42)),
])
# fit the model on the training data
model.fit(X_train_f, y_train)




# predict on dev and test sets
y_dev_pred  = model.predict(X_dev_f)
y_test_pred = model.predict(X_test_f)




def print_report(y_true, y_pred, split_name):
    # compute report and print accuracy, precision, recall, f1-score for each class
    report = classification_report(y_true, y_pred, target_names=["ham", "spam"],
                                   output_dict=True)
    print(f"=== {split_name} ===")
    # print accuracy and then precision, recall, f1-score for each class in a nice format
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"{'':20s} {'precision':>10} {'recall':>10} {'f1-score':>10}")
    # loop through the classes and print their metrics
    for cls in ["ham", "spam"]:
        r = report[cls]
        print(f"  {cls:18s} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f}")
    print()


# print the results for dev and test sets
print_report(y_dev,  y_dev_pred,  "Dev Set")
print_report(y_test, y_test_pred, "Test Set")