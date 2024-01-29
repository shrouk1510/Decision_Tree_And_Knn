import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv(r'E:\4thLevel\Ml\diabetes.csv')

# Data Preprocessing
# Normalize each feature column separately using Min-Max Scaling
# Apply Min-Max Scaling to each feature except the 'Outcome' column
normalized_data = (data - data.min()) / (data.max() - data.min())

# Manually split the data into training and testing sets using train_size
train_size = int(0.7 * len(data))
train_data = normalized_data[:train_size]
test_data = normalized_data[train_size:]

# Training set
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']

# Testing set
X_test = test_data.drop('Outcome', axis=1)
y_test = test_data['Outcome']

# Define the KNN algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(x_test, X_train.iloc[i])
        distances.append((distance, y_train.iloc[i]))

    distances = sorted(distances)[:k]
    
    # Distance-Weighted Voting
    counts = {}
    for i in range(len(distances)):
        distance, label = distances[i]
        weight = 1 / (distance + 1e-5)  # Avoid division by zero
        if label in counts:
            counts[label] += weight
        else:
            counts[label] = weight
    
    prediction = max(counts, key=counts.get)
    return prediction

# Perform multiple iterations of KNN for different 'K' values
k_values = [2, 3, 4, 5, 7]  # You can use other 'K' values as well

for k in k_values:
    correct = 0
    for i in range(len(X_test)):
        pred = predict(X_train, y_train, X_test.iloc[i], k)
        if pred == y_test.iloc[i]:
            correct += 1
    accuracy = correct / len(X_test) * 100
    print(f'k value: {k}, "Number of correctly classified instances: {correct},'
          f'Total number of instances: {len(X_test)}, Accuracy: {accuracy:.2f}%')