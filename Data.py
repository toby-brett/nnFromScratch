from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Train set shape:", X_train.shape, y_train.shape)
#print("Test set shape:", X_test.shape, y_test.shape)

# Function to create batches
def create_batches(X, y, batch_size):
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split into batches
    n_batches = int(np.ceil(X.shape[0] / batch_size))
    batches = [
        (X[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
        for i in range(n_batches)
    ]
    return batches

# Define batch size
batch_size = 32

# Create batches for train and test sets
train_batches = create_batches(X_train, y_train, batch_size)

train_batches.pop()

#print(train_batches[0][0]) # train_batches[batch][data=0 labels=1]

