from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model_utils import predict_species

app = FastAPI(
    title="Iris Species Prediction API",
    description="Predicts the Iris flower species from four measurements (cm).",
    version="1.0.0",
)


class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, le=10, examples=[5.1])
    sepal_width: float = Field(..., gt=0, le=10, examples=[3.5])
    petal_length: float = Field(..., gt=0, le=10, examples=[1.4])
    petal_width: float = Field(..., gt=0, le=10, examples=[0.2])


class PredictionOutput(BaseModel):
    predicted_species: str


@app.get("/")
def root():
    return {"message": "Iris Prediction API is running. See /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    species = predict_species(
        data.sepal_length, data.sepal_width, data.petal_length, data.petal_width
    )
    return {"predicted_species": species}
