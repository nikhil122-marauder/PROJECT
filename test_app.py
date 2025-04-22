# test_app.py

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app import app, make_basic_regressors

client = TestClient(app)

def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "forecast/tube/full2024" in body["message"]
    assert "forecast/tube/basic" in body["message"]

def test_basic_regressors_helper():
    # Build a small series of dates: Jan, Jun, Oct 2024
    dates = pd.to_datetime(["2024-01-01", "2024-06-15", "2024-10-01"])
    df = make_basic_regressors(dates)
    # It should have exactly these three columns
    assert list(df.columns) == ["ds", "is_weekend", "season"]
    # 2024‑01‑01 is Monday → not weekend, season=3 (winter)
    assert df.loc[0, "is_weekend"] == False
    assert df.loc[0, "season"] == 3
    # 2024‑06‑15 is Saturday → weekend, season=1 (summer)
    assert df.loc[1, "is_weekend"] == True
    assert df.loc[1, "season"] == 1

def test_tube_full2024_endpoint():
    resp = client.get("/forecast/tube/full2024")
    assert resp.status_code == 200
    data = resp.json()["forecast_tube_full2024"]
    assert isinstance(data, list)
    # Every item must have a 2024 date and a yhat
    for item in data:
        ds = pd.to_datetime(item["ds"])
        assert ds.year == 2024
        assert "yhat" in item

def test_bus_full2024_endpoint():
    resp = client.get("/forecast/bus/full2024")
    assert resp.status_code == 200
    data = resp.json()["forecast_bus_full2024"]
    assert isinstance(data, list)
    for item in data:
        ds = pd.to_datetime(item["ds"])
        assert ds.year == 2024
        assert "yhat" in item

@pytest.mark.parametrize("endpoint,key", [
    ("/forecast/tube/basic", "forecast_tube_basic"),
    ("/forecast/bus/basic",  "forecast_bus_basic")
])
def test_basic_dynamic_endpoints(endpoint, key):
    # Request a 5‑day forecast starting 2024‑07‑01
    payload = {"start_date": "2024-07-01T00:00:00", "periods": 5}
    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 200
    data = resp.json()[key]
    # Should be a list of length 5
    assert isinstance(data, list)
    assert len(data) == 5

    # Dates should be monotonically increasing from 2024‑07‑01
    dates = [pd.to_datetime(rec["ds"]) for rec in data]
    assert dates == sorted(dates)
    assert dates[0] == pd.to_datetime("2024-07-01")

    # Each record must have a yhat
    assert all("yhat" in rec for rec in data)
