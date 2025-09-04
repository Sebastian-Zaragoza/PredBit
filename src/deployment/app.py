import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from src.utils.config import load_config

config = load_config()
model_path = project_root / config["paths"]["model_dir"] / "random_forest_anxiety_level_pipeline.pkl"
try:
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError("Model not found.")

app = FastAPI(
    title="PredBit - API Habit Prediction",
    description="API to predict who can keep a habit based on multiple features (actually it predicts anxiety levels, that is part of the four features selected to enhance this predictor)."
)

class PredictionPayload(BaseModel):
    daily_screen_time_min: float
    notification_count: int
    social_media_time_min: float

@app.get("/anxiety-levels")
def read_root():
    return {"message": "API to predict anxiety levels."}

@app.post("/predict")
def predict_habit(payload: PredictionPayload):
    input_data = pd.DataFrame([payload.dict()])
    prediction = model_pipeline.predict(input_data)[0]

    result = "Your anxiety levels are low" if prediction == 1 else "Your anxiety levels are high"
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)