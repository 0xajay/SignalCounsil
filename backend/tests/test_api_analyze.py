import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from tests.fixtures.market_data import get_ohlcv_models

client = TestClient(app)

@patch("main.validate_symbol", return_value=True)
@patch("main.get_ohlcv")
@patch("main.get_latest_price")
def test_api_analyze_endpoint(mock_get_price, mock_get_ohlcv, mock_validate):
    mock_get_ohlcv.return_value = get_ohlcv_models()
    mock_get_price.return_value = 150.25

    response = client.post("/api/v1/analyze", json={
        "symbol": "AAPL",
        "market": "US",
        "timeframe": "1d"
    })

    assert response.status_code == 200
    data = response.json()

    # Verify all required output fields from specification
    assert "directional_bias" in data
    assert data["directional_bias"] in ["BULLISH", "BEARISH", "NEUTRAL"]
    assert "evidence_strength" in data
    assert data["evidence_strength"] in ["STRONG", "MODERATE", "WEAK", "INSUFFICIENT"]
    assert "setup_status" in data
    assert data["setup_status"] in ["CONFIRMED", "UNCONFIRMED", "CONFLICTED", "INVALID_DATA"]
    assert "backtest_status" in data
    assert data["backtest_status"] in ["PASS", "WARNING", "FAIL", "INSUFFICIENT_DATA"]
    assert "thesis_status" in data
    assert data["thesis_status"] in ["INTACT", "INVALIDATED", "NOT_ESTABLISHED"]

    assert "supporting_evidence" in data
    assert "opposing_evidence" in data
    assert "neutral_evidence" in data
    assert "evidence_dependencies" in data
    assert "independent_evidence_families" in data
    assert "historical_setup_results" in data
    assert "confirmation_conditions" in data
    assert "invalidation_conditions" in data
    assert "data_quality_warnings" in data
    assert "explanation" in data
    assert "verdict" in data
    assert "reconciliation_status" in data
    assert "instrument" in data
    assert "data_quality" in data
    assert "evidence_graph" in data
    assert "historical_comparison" in data
    assert "trade_geometry" in data

    # Verify historical setup results structure
    hist = data["historical_setup_results"]
    assert "setup_definition" in hist
    assert "sample_count" in hist
    assert "win_rate" in hist
    assert "profit_factor" in hist
    assert "max_drawdown" in hist
    assert "net_return" in hist
    assert "transaction_cost_pct" in hist

    # Verify graph and geometry structure
    assert "nodes" in data["evidence_graph"]
    assert "edges" in data["evidence_graph"]
    assert "risk_stop_rule" in data["trade_geometry"]
    assert "structural_stop_rule" in data["trade_geometry"]

    # Verify audit fields in historical_comparison
    hc = data["historical_comparison"]
    assert "matching_status" in hc
    assert "historical_strategy_validation" in hc
    assert "data_overlap_disclosure" in hc
    assert "mean_raw_market_return_pct" in hc
    assert "mean_per_trade_return_pct" in hc
    assert "directional_family_alignment_formula" in data
    assert "directional_family_alignment_participating" in data

    # Verify walk-forward, portfolio simulation, and drawdown audit records
    assert "portfolio_simulation" in hc
    assert hc["portfolio_simulation"]["max_concurrent_positions"] <= 5
    assert "walk_forward_evaluation" in hc
    assert "regime_breakdowns" in hc["walk_forward_evaluation"]
    assert "drawdown_audit_records" in hist
    assert len(hist["drawdown_audit_records"]) >= 4
    assert any(r["source_dataset"] == "FULL_STRATEGY_BACKTEST" for r in hist["drawdown_audit_records"])
    assert any(r["source_dataset"] == "PORTFOLIO_SIMULATION" for r in hist["drawdown_audit_records"])

@patch("main.validate_symbol", return_value=True)
@patch("main.get_ohlcv")
@patch("main.get_latest_price")
def test_api_analyze_with_market_normalization(mock_get_price, mock_get_ohlcv, mock_validate):
    mock_get_ohlcv.return_value = get_ohlcv_models()
    mock_get_price.return_value = 2950.0

    # User sends clean ticker "RELIANCE" and market "NSE" without any .NS suffix
    response = client.post("/api/v1/analyze", json={
        "symbol": "RELIANCE",
        "market": "NSE",
        "timeframe": "1d"
    })

    assert response.status_code == 200
    data = response.json()

    # Response should keep clean symbol and market for the UI
    assert data["symbol"] == "RELIANCE"
    assert data["market"] == "NSE"

    # Adapter should have received normalized ticker RELIANCE.NS for Yahoo Finance
    args, _ = mock_get_ohlcv.call_args_list[0]
    assert args[0] == "RELIANCE.NS"

    # Instrument metadata reflects the market and canonical provider symbol
    assert data["instrument"]["symbol"] == "RELIANCE"
    assert data["instrument"]["canonical_symbol"] == "RELIANCE.NS"
    assert data["instrument"]["exchange"] == "NSE"
    assert data["instrument"]["currency"] == "INR"

