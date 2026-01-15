from fastapi import FastAPI
from pydantic import BaseModel, Field
from model import predict

app = FastAPI(title="ML Prediction API")

class InputData(BaseModel):
    feature1: float = Field(..., example=5)
    feature2: float = Field(..., example=10)

class Prediction(BaseModel):
    prediction: float

@app.post("/predict", response_model=Prediction)
def get_prediction(data: InputData):
    result = predict([data.feature1, data.feature2])
    return {"prediction": result}
