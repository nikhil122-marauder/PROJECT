import mlflow
import pandas as pd
import os
from datetime import datetime, timedelta

# Configuration
THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.08"))  # 8% MAPE
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

def fetch_recent_mape(experiment_name="Default"):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    # Find the experiment
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    # Compute the cutoff timestamp
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    # Search all runs since cutoff
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.start_time >= {int(cutoff.timestamp() * 1000)}"
    )
    # Extract all MAPE metrics
    mapes = []
    for run in runs:
        mape = run.data.metrics.get("MAPE") or run.data.metrics.get("mape") or run.data.metrics.get("smape")
        if mape is not None:
            mapes.append(mape)
    if not mapes:
        return None
    return sum(mapes) / len(mapes)

def main():
    avg_mape = fetch_recent_mape()
    if avg_mape is None:
        print("No recent MAPE data; skipping retrain.")
        exit(0)
    print(f"Average MAPE over last {LOOKBACK_DAYS} days: {avg_mape:.4f}")
    drift = avg_mape > THRESHOLD
    # Emit GitHub Actions output syntax
    print(f"drift_detected={str(drift).lower()}")

if __name__ == "__main__":
    main()