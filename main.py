"""How to run the script:
pip install -r requirements.txt
python main.py
"""

from sklearn.datasets import make_classification
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


X, Y = make_classification(
    n_features=3,
    n_redundant=0,
    random_state=1,
    n_informative=3,
    n_clusters_per_class=1,
    n_classes=3
)

TRAIN_PROPORTION = .75
TRAIN_SIZE = int(len(X) * TRAIN_PROPORTION)
X_train, y_train = X[:TRAIN_SIZE], Y[:TRAIN_SIZE]
X_test, y_test = X[TRAIN_SIZE:], Y[TRAIN_SIZE:]


def euclidean_distance(vector_1: list, vector_2: list) -> float:
    # https://en.wikipedia.org/wiki/Euclidean_distance
    return sum([(v1 - v2)**2 for v1, v2 in zip(vector_1, vector_2)]) ** .5


def knn_evaluate(X_train, y_train, X_test, y_test, k):
    results = []
    for test_vector, test_label in zip(X_test, y_test):
        distance_data = []
        for train_vector, train_label in zip(X_train, y_train):
            distance = euclidean_distance(test_vector, train_vector)
            data = {"label": train_label, "distance": distance}
            distance_data.append(data)
        sorted_data = sorted(distance_data, key=lambda d: d['distance'])
        top_k_labels = [data["label"] for data in sorted_data][:k]
        prediction = max(set(top_k_labels), key=top_k_labels.count)
        results.append(
            {
                "truth": test_label,
                "prediction": prediction,
                "correct": test_label == prediction
            }
        )
    df = pd.DataFrame(results)
    acc = sum(df["correct"]) / len(df["correct"])
    return df, acc


K = 9
eval_df, acc = knn_evaluate(X_train, y_train, X_test, y_test, K)
print("KNN model from scratch acc: ", acc)


model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)
predictions_from_sklearn = model.predict(X_test)
sklearn_acc = sum(predictions_from_sklearn == y_test) / len(y_test)
print("KNN model from sklearn acc: ", sklearn_acc)


matches = []
for A, B in zip(list(predictions_from_sklearn), list(eval_df["prediction"])):
    matches.append(A == B)
if all(matches):
    print("KNN model from scratch matches sklearn results ✅")
else:
    print("KNN model from scratch matches does not match sklearn results ⚠️")
