import pandas as pd
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load data
data = pd.read_csv("data.csv")
X = data[["feature1", "feature2"]]
y = data["label"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model trained and exported to model.onnx")
