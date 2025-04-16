from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn

# ---------------------------
# Load Prophet Models from Pickle Files
# ---------------------------
# Replace the file names with the actual paths where your models are saved.
with open("tube_prophet_model.pkl", "rb") as f:
    tube_model = pickle.load(f)

with open("bus_prophet_model.pkl", "rb") as f:
    bus_model = pickle.load(f)

# ---------------------------
# FastAPI Application Setup
# ---------------------------
app = FastAPI(
    title="Transport Forecasting API",
    description="API serving forecasts for both Tube and Bus journey counts using Prophet models (loaded from pickle).",
    version="1.0",
)

# Pydantic model for forecasting request
class ForecastRequest(BaseModel):
    future_periods: int  # Number of days to forecast

# Root endpoint providing a welcome message
@app.get("/")
def root():
    return {"message": "Welcome to the Transport Forecasting API. Use /forecast/tube or /forecast/bus endpoints."}

# Endpoint for Tube journey forecasts
@app.post("/forecast/tube")
def forecast_tube(request: ForecastRequest):
    future_periods = request.future_periods
    # Create future DataFrame for Tube model with daily frequency
    future_df = tube_model.make_future_dataframe(periods=future_periods, freq="D")
    forecast_df = tube_model.predict(future_df)
    # Only select the forecasted portion
    forecast_results = forecast_df[forecast_df["ds"] >= future_df["ds"].iloc[-future_periods]]
    # Prepare output: list of dicts with date and forecasted value
    result = forecast_results[["ds", "yhat"]].tail(future_periods).to_dict(orient="records")
    return {"forecast_tube": result}

# Endpoint for Bus journey forecasts
@app.post("/forecast/bus")
def forecast_bus(request: ForecastRequest):
    future_periods = request.future_periods
    # Create future DataFrame for Bus model with daily frequency
    future_df = bus_model.make_future_dataframe(periods=future_periods, freq="D")
    forecast_df = bus_model.predict(future_df)
    # Only select the forecasted portion
    forecast_results = forecast_df[forecast_df["ds"] >= future_df["ds"].iloc[-future_periods]]
    # Prepare output: list of dicts with date and forecasted value
    result = forecast_results[["ds", "yhat"]].tail(future_periods).to_dict(orient="records")
    return {"forecast_bus": result}

# To run this application locally, uncomment the following lines:
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    
# Note: You can run this FastAPI app using the command: uvicorn app:app --reload