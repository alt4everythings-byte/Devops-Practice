import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data.csv")
X = data[["feature1", "feature2"]].values
y = data["label"].values

# Train sklearn model
sk_model = LinearRegression()
sk_model.fit(X, y)

# Load ONNX model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

sample = X[:1].astype(np.float32)

# Benchmark sklearn inference
start = time.time()
for _ in range(1000):
    sk_model.predict(sample)
sk_time = time.time() - start

# Benchmark ONNX inference
start = time.time()
for _ in range(1000):
    session.run(None, {input_name: sample})
onnx_time = time.time() - start

print(f"scikit-learn inference time: {sk_time:.6f} seconds")
print(f"ONNX Runtime inference time: {onnx_time:.6f} seconds")
