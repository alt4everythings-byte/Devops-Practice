from sklearn.linear_model import LinearRegression
import numpy as np

# Train a simple model
X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([3,5,7,9])

model = LinearRegression()
model.fit(X, y)

def predict(features):
    return model.predict([features])[0]
