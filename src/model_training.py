import os
from datetime import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# -----------------------
# Paths (all outputs saved to BASE_DIR)
# -----------------------
BASE_DIR = r"C:\Users\acer\Desktop\PBL 5th sem"
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv(DATA_PATH)

# optional mappings (adjust if your CSV differs)
if "Sex" in df.columns:
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
if "ChestPainType" in df.columns:
    df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
if "RestingECG" in df.columns:
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
if "ExerciseAngina" in df.columns:
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
if "ST_Slope" in df.columns:
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
if "HeartDisease" in df.columns:
    df['HeartDisease'] = df['HeartDisease'].astype(int)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# scaler for models that need scaled input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Models to evaluate
# -----------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

# results bookkeeping
results = {}
trained_order = []

def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Utility: save per-model confusion matrix image and classification report text (saved to BASE_DIR)
def save_confusion_and_report(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    ts = _timestamp()
    cm_fname = os.path.join(BASE_DIR, f"{name}_confusion_{ts}.png")
    # plot confusion matrix
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i,j] > (cm.max()/2) else "black")
    plt.tight_layout()
    plt.savefig(cm_fname, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {cm_fname}")

    # save classification report
    report = classification_report(y_true, y_pred, digits=4)
    rep_fname = os.path.join(BASE_DIR, f"{name}_report_{ts}.txt")
    with open(rep_fname, "w") as f:
        f.write(f"Classification report for {name}\n\n")
        f.write(report)
    print(f"Saved classification report: {rep_fname}")

# Utility: save a combined bar chart for accuracy/precision/recall/f1 (saved to BASE_DIR)
def save_metrics_chart(metrics_dict, highlight=None):
    """
    metrics_dict: {model_name: {"accuracy":.., "precision":.., "recall":.., "f1":..}, ...}
    """
    dfm = pd.DataFrame(metrics_dict).T  # models as rows
    # Ensure all required columns exist (if a model missing, fill with zeros)
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col not in dfm.columns:
            dfm[col] = 0.0
    dfm = dfm[["accuracy", "precision", "recall", "f1"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    dfm.plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Models: accuracy / precision / recall / f1 (so far)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.xticks(rotation=25, ha='right')

    # annotate each bar with numeric value (positioning works well for a few models)
    n_models = len(dfm)
    width = 0.2
    for i, col in enumerate(dfm.columns):
        vals = dfm[col].values
        for j, val in enumerate(vals):
            # compute x position similar to grouped bars
            x = j + (i - 1.5) * width
            ax.text(x, val + 0.01, f"{val:.3f}", fontsize=8, ha='center')

    # highlight a model by drawing a rectangle (if provided)
    if highlight and highlight in dfm.index:
        idx = list(dfm.index).index(highlight)
        bbox = dict(edgecolor='red', linewidth=2, fill=False)
        ax.add_patch(plt.Rectangle((idx - 0.45, 0), 0.9, 1.02, **bbox))

    ts = _timestamp()
    fname = os.path.join(BASE_DIR, f"metrics_combined_{ts}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved combined metrics chart: {fname}")
    return fname

# Train & evaluate
for name, model in models.items():
    print(f"\nTraining {name} ...")
    # choose scaled vs unscaled input
    if name in ["LogisticRegression", "SVM", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    trained_order.append(name)

    print(f"{name} -> accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

    # save confusion matrix + classification report directly to BASE_DIR
    save_confusion_and_report(name, y_test, y_pred)

    # save a combined chart for all models trained so far (single file in BASE_DIR)
    sub_metrics = {k: results[k] for k in trained_order}
    save_metrics_chart(sub_metrics, highlight=name)

# Save best model (by f1)
best_model_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_model_name]
best_metrics = results[best_model_name]

# Persist model and scaler (to BASE_DIR)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)
print(f"\nSaved best model to: {MODEL_PATH} ({best_model_name})")

if best_model_name in ["LogisticRegression", "SVM", "KNN"]:
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {SCALER_PATH}")

# Save overall metrics to CSV in BASE_DIR
metrics_df = pd.DataFrame(results).T
metrics_csv = os.path.join(BASE_DIR, "models_metrics.csv")
metrics_df.to_csv(metrics_csv, index=True)
print(f"Saved all model metrics to: {metrics_csv}")

# Save a final combined chart highlighting best model (stable name)
final_chart = save_metrics_chart(results, highlight=best_model_name)
final_chart_stable = os.path.join(BASE_DIR, "models_metrics_summary.png")
try:
    os.replace(final_chart, final_chart_stable)
    print(f"Final summary chart saved as: {final_chart_stable}")
except Exception:
    print("Could not rename final chart; it should still be in BASE_DIR.")

print(f"\nBest model by F1: {best_model_name} -> {best_metrics}")
print("Done.")
