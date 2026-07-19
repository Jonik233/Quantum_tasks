import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from inference import load_model, predict_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler: Loads heavy ML models into memory on startup 
    and cleans them up on shutdown.
    """
    LOCAL_MODEL_DIR = os.environ.get("MODEL_DIR", "./model_artifacts")

    try:
        app.state.model_dict = load_model(LOCAL_MODEL_DIR)
        print("=> FastAPI lifespan: Model successfully loaded into application state.")
    except Exception as e:
        print(f"Failed to load model from {LOCAL_MODEL_DIR}. Error: {e}")
        raise RuntimeError("Application startup failed due to model loading error.")

    yield

    # Clean up memory on shutdown
    print("=> FastAPI lifespan: Cleaning up model memory...")
    model_dict = getattr(app.state, "model_dict", None)
    if model_dict:
        del model_dict
        del app.state.model_dict


app = FastAPI(lifespan=lifespan, title="Mountain NER API")


class QueryPayload(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Text to extract mountains from.")


@app.post("/predict")
def predict(payload: QueryPayload):
    try:
        results = predict_fn(payload.query, app.state.model_dict)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))