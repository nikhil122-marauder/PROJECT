# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import pickle
import mlflow


# 1. Load all four models (pickle files)
with open("tube_prophet_model.pkl", "rb") as f:
    tube_model_full = pickle.load(f)
with open("bus_prophet_model.pkl", "rb") as f:
    bus_model_full = pickle.load(f)
with open("tube_prophet_model_future.pkl", "rb") as f:
    tube_model_basic = pickle.load(f)
with open("bus_prophet_model_future.pkl", "rb") as f:
    bus_model_basic = pickle.load(f)

# 2. Load regressors for full models (must cover 2024-01-01 to 2024-12-31)
regressors = (
    pd.read_csv("regressors.csv", parse_dates=["ds"])
      .set_index("ds")
)

# Helper to compute basic calendar features
def make_basic_regressors(dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"ds": dates})
    df["is_weekend"] = df["ds"].dt.weekday >= 5
    # Meteorological seasons: Mar–May=0, Jun–Aug=1, Sep–Nov=2, Dec–Feb=3
    m = df["ds"].dt.month
    df["season"] = (
        0 * m.isin([3,4,5]) +
        1 * m.isin([6,7,8]) +
        2 * m.isin([9,10,11]) +
        3 * m.isin([12,1,2])
    )
    return df

# 3. FastAPI setup
app = FastAPI(
    title="Transport Forecasting API",
    description="Prophet forecasts for Tube & Bus journeys (2024-full or basic-calendar).",
    version="1.0",
)

class RangeRequest(BaseModel):
    start_date: datetime  # e.g. "2024-05-01"
    periods: int          # number of days to forecast

@app.get("/")
def root():
    return {
        "message": "Use /forecast/tube/full2024, /forecast/bus/full2024, "
                   "/forecast/tube/basic, /forecast/bus/basic"
    }

# 4a. Full‐regressor endpoints (fixed to 2024)
@app.get("/forecast/tube/full2024")
def forecast_tube_full2024():
    # slice regressors to 2024
    df_regr = regressors.loc["2024-01-01":"2024-12-31"].reset_index()
    # future df with exactly len(df_regr) days beyond training
    future = tube_model_full.make_future_dataframe(periods=len(df_regr), freq="D")
    future = future.merge(
        df_regr[["ds","t1","hum","wind_speed","is_holiday","is_weekend","season"]],
        on="ds", how="inner"
    )
    forecast = tube_model_full.predict(future)
    out = forecast[forecast["ds"].dt.year == 2024][["ds","yhat"]]
    # 4) Log to MLflow
    with mlflow.start_run(run_name="api_tube_full", nested=True):
        mlflow.set_tag("endpoint", "tube_full")
        # summary stats
        y = forecast["yhat"]
        mlflow.log_metric("yhat_mean",    float(y.mean()))
        mlflow.log_metric("yhat_min",     float(y.min()))
        mlflow.log_metric("yhat_max",     float(y.max()))
        # optionally log full predictions
        mlflow.log_dict(out, "predictions.json")
    return {"forecast_tube_full2024": out.to_dict(orient="records")}

@app.get("/forecast/bus/full2024")
def forecast_bus_full2024():
    df_regr = regressors.loc["2024-01-01":"2024-12-31"].reset_index()
    future = bus_model_full.make_future_dataframe(periods=len(df_regr), freq="D")
    future = future.merge(
        df_regr[["ds","t1","hum","wind_speed","is_holiday","is_weekend","season"]],
        on="ds", how="inner"
    )
    forecast = bus_model_full.predict(future)
    out = forecast[forecast["ds"].dt.year == 2024][["ds","yhat"]]
    # 4) Log to MLflow
    with mlflow.start_run(run_name="api_bus_full", nested=True):
        mlflow.set_tag("endpoint", "bus_full")
        # summary stats
        y = forecast["yhat"]
        mlflow.log_metric("yhat_mean",    float(y.mean()))
        mlflow.log_metric("yhat_min",     float(y.min()))
        mlflow.log_metric("yhat_max",     float(y.max()))
        # optionally log full predictions
        mlflow.log_dict(out, "predictions.json")
    return {"forecast_bus_full2024": out.to_dict(orient="records")}

# 4b. Basic‐regressor endpoints (dynamic dates)

@app.post("/forecast/tube/basic")
def forecast_tube_basic(req: RangeRequest):
    # 1) Build exactly the dates you want
    dates = pd.date_range(req.start_date, periods=req.periods, freq="D")
    df_basic = make_basic_regressors(dates)
    # 2) Predict directly on df_basic (it has ds, is_weekend, season)
    forecast = tube_model_basic.predict(df_basic)
    # 3) Return those same req.periods rows
    out = forecast[["ds", "yhat"]].to_dict(orient="records")
    # 4) Log to MLflow
    with mlflow.start_run(run_name="api_tube_basic", nested=True):
        mlflow.set_tag("endpoint", "tube_basic")
        mlflow.log_param("periods", req.periods)
        # summary stats
        y = forecast["yhat"]
        mlflow.log_metric("yhat_mean",    float(y.mean()))
        mlflow.log_metric("yhat_min",     float(y.min()))
        mlflow.log_metric("yhat_max",     float(y.max()))
        # optionally log full predictions
        mlflow.log_dict(out, "predictions.json")
    return {"forecast_tube_basic": out}

@app.post("/forecast/bus/basic")
def forecast_bus_basic(req: RangeRequest):
    dates = pd.date_range(req.start_date, periods=req.periods, freq="D")
    df_basic = make_basic_regressors(dates)
    forecast = bus_model_basic.predict(df_basic)
    out = forecast[["ds", "yhat"]].to_dict(orient="records")
    # 4) Log to MLflow
    with mlflow.start_run(run_name="api_bus_basic", nested=True):
        mlflow.set_tag("endpoint", "bus_basic")
        mlflow.log_param("periods", req.periods)
        # summary stats
        y = forecast["yhat"]
        mlflow.log_metric("yhat_mean",    float(y.mean()))
        mlflow.log_metric("yhat_min",     float(y.min()))
        mlflow.log_metric("yhat_max",     float(y.max()))
        # optionally log full predictions
        mlflow.log_dict(out, "predictions.json")
    return {"forecast_bus_basic": out}


\
