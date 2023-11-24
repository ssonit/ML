import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = open('ring.dat').read().split('\n')

# Loại bỏ metadata
data = [line for line in data if line and not line.startswith('@')]


# Extract values and labels from each line
samples = []
labels = []
for line in data:
  values = [float(x) for x in line.split(',')[:-1]]
  label = int(line.split(',')[-1])
  samples.append(values)
  labels.append(label)

# Convert to numpy arrays
samples = np.array(samples)
labels = np.array(labels)
X = samples
y = labels


dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=25)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracies = []
precisions = []
recalls = []
f1_scores = []

# Thực hiện k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Huấn luyện mô hình Decision Tree
    dt_classifier.fit(X_train, y_train)

    # Dự đoán
    y_pred = dt_classifier.predict(X_test)

    # Tính toán các độ đo đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Lưu trữ kết quả
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# In kết quả trung bình của các độ đo đánh giá
average_accuracy = np.mean(accuracies)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_f1 = np.mean(f1_scores)


print(f'Average Accuracy: %.3f%%' % (average_accuracy * 100))
print(f'Average Precision: %.3f%%' % (average_precision * 100))
print(f'Average Recall: %.3f%%' % (average_recall * 100))
print(f'Average F1 Score: %.3f%%' % (average_f1 * 100))