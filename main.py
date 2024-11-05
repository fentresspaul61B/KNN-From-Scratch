"""How to run the script:
pip install -r requirements.txt
python main.py
"""

from sklearn.datasets import make_classification
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------------------------------------------------------
# GENERATING SYNTHETIC DATA
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

X, Y = make_classification(
    n_features=3,
    n_redundant=0,
    random_state=1,
    n_informative=3,
    n_clusters_per_class=1,
    n_classes=3
)

# -----------------------------------------------------------------------------
# TRAIN TEST SPLIT

TRAIN_PROPORTION = .75
TRAIN_SIZE = int(len(X) * TRAIN_PROPORTION)
X_train, y_train = X[:TRAIN_SIZE], Y[:TRAIN_SIZE]
X_test, y_test = X[TRAIN_SIZE:], Y[TRAIN_SIZE:]

# -----------------------------------------------------------------------------
# COMPUTING VECTOR DISTANCES HELPER


def euclidean_distance(vector_1, vector_2):
    # https://en.wikipedia.org/wiki/Euclidean_distance
    return sum([(v1 - v2)**2 for v1, v2 in zip(vector_1, vector_2)]) ** .5


def create_distance_matrix(X_train: list, X_test: list) -> pd.DataFrame:
    distances = []
    for v1 in X_test:
        row = []
        for v2 in X_train:
            row.append(euclidean_distance(v1, v2))
        distances.append(row)
    return pd.DataFrame(distances)

# -----------------------------------------------------------------------------
# EVALUATE TEST DATA HELPER


def knn_evaluate(distances: pd.DataFrame, k: int) -> pd.DataFrame:
    eval_data = []
    for i, row in enumerate(distances.iterrows()):
        distance_data = []
        for j, value in enumerate(row[1]):
            data = {"label": y_train[j], "distance": value}
            distance_data.append(data)
        sorted_distances = sorted(distance_data, key=lambda d: d['distance'])
        labels = [data["label"] for data in sorted_distances][:k]
        prediction = max(set(labels), key=labels.count)
        truth = y_test[i]
        eval_data.append(
            {
                "truth": truth,
                "prediction": prediction,
                "correct": truth == prediction
            }
        )
    df = pd.DataFrame(eval_data)
    acc = sum(df["correct"]) / len(df["correct"])
    return df, acc

# -----------------------------------------------------------------------------
# RUNNING EXPERIMENT


K = 9
distances = create_distance_matrix(X_train, X_test)
eval_df, model_from_scratch_acc = knn_evaluate(distances, K)
print("KNN Model from scratch acc: ", model_from_scratch_acc)

# -----------------------------------------------------------------------------
# SKLEARN EXPERIMENT


model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)
predictions_from_sklearn = model.predict(X_test)
sklearn_acc = sum(predictions_from_sklearn == y_test) / len(y_test)
print("KNN Model from sklearn acc: ", sklearn_acc)

# -----------------------------------------------------------------------------
# COMPARING RESULTS

matches = []
for A, B in zip(list(predictions_from_sklearn), list(eval_df["prediction"])):
    matches.append(A == B)

if all(matches):
    print("KNN Model from scratch matches sklearn results ✅")
else:
    print("KNN Model from scratch matches does not match sklearn results ⚠️")
