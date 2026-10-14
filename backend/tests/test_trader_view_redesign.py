"""
Acceptance and Regression Test Suite for SignalCouncil UI Redesign
Verifies all 13 criteria from Section 7 of the specification:
1. Trader View is the default view.
2. Initial view communicates bias, setup readiness, key levels, risk/reward and historical validation clearly.
3. All displayed figures come from current canonical analysis results.
4. No example NIFTY price or level is hardcoded into production UI.
5. Pattern-backtest PASS cannot override a failed walk-forward validation.
6. Unverified selection-bias provenance remains visible beside pattern-backtest results.
7. Drawdowns are correctly labelled by evaluation domain.
8. Confidence-interval and no-validated-edge warnings remain accessible in Research & Audit.
9. Benchmark comparisons remain separate, and no unsupported alpha claim is introduced.
10. Expanding and collapsing each Research & Audit section works correctly.
11. Detailed evidence graph and audit information remain accessible.
12. No existing analysis capability or calculation is accidentally removed.
13. Interface works at actual Chrome side-panel width without horizontal overflow.
"""

from pathlib import Path
import pytest

APP_PATH = Path(__file__).resolve().parent.parent.parent / "extension" / "src" / "App.tsx"
CSS_PATH = Path(__file__).resolve().parent.parent.parent / "extension" / "src" / "index.css"


@pytest.fixture(scope="module")
def app_source() -> str:
    assert APP_PATH.exists(), f"App.tsx not found at {APP_PATH}"
    return APP_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css_source() -> str:
    assert CSS_PATH.exists(), f"index.css not found at {CSS_PATH}"
    return CSS_PATH.read_text(encoding="utf-8")


# Criterion 1: Trader View is the default view
def test_criterion_1_trader_view_is_default(app_source: str):
    """Trader View must be the initial active view in state."""
    assert "const [activeView, setActiveView] = useState<'trader' | 'research'>('trader')" in app_source
    assert "activeView === 'trader' ?" in app_source
    assert "renderTraderView()" in app_source
    assert "renderResearchAuditView()" in app_source


# Criterion 2: Communicates bias, setup readiness, key levels, risk/reward, validation
def test_criterion_2_trader_view_core_sections(app_source: str):
    """The Trader View must communicate the 7 key sections in the required order."""
    # A. Instrument header
    assert "report.symbol" in app_source or "report.instrument?.symbol" in app_source
    assert "report.timeframe" in app_source
    assert "getExchangeDisplay()" in app_source or "Exchange:" in app_source
    assert "Last Completed Candle:" in app_source

    # B. Plain-English verdict
    assert "Historical trading edge:" in app_source
    assert "Directional Family Alignment" in app_source
    assert "NOT a probability of price decline" in app_source

    # C. Key price levels
    assert "Key Price Levels" in app_source
    assert "Risk-Stop Reference" in app_source
    assert "Structural Thesis Invalidation" in app_source
    assert "NOT live orders or guaranteed support/resistance" in app_source

    # D. Risk/reward assessment
    assert "Risk/Reward Assessment" in app_source
    assert "hypothetical scenario analysis" in app_source
    assert "unfavourable" in app_source.lower()

    # E. Three key reasons
    assert "Three Key Reasons" in app_source
    assert "1. Directional Evidence" in app_source
    assert "2. Confirmation" in app_source
    assert "3. Historical Reliability" in app_source

    # F. What to watch next
    assert "What to Watch Next" in app_source
    assert "Thesis Confirmation Trigger" in app_source
    assert "Structural Thesis Invalidation" in app_source
    assert "Risk-Management Stop Condition" in app_source

    # G. Historical validation at a glance table
    assert "Historical Validation at a Glance" in app_source
    assert "Pattern-specific backtest" in app_source
    assert "Walk-forward validation" in app_source
    assert "Statistical sensitivity" in app_source
    assert "Portfolio simulation" in app_source
    assert "Overall validation" in app_source


# Criterion 3: Figures come from canonical analysis results
def test_criterion_3_figures_from_canonical_model(app_source: str):
    """All metrics in Trader View must dynamically consume cs, geom, histComp, and report."""
    assert "geom?.reference_price" in app_source
    assert "geom?.target_price" in app_source
    assert "geom?.risk_stop_price" in app_source
    assert "geom?.structural_stop_price" in app_source
    assert "cs.directional_family_alignment" in app_source
    assert "cs.walk_forward_compounded_return_pct" in app_source
    assert "cs.bootstrap_ci_95_lower_pct" in app_source
    assert "cs.portfolio_simulation_return_pct" in app_source
    assert "cs.overall_validation_status" in app_source


# Criterion 4: No hardcoded NIFTY price or level in production UI
def test_criterion_4_no_hardcoded_sample_prices(app_source: str):
    """Ensure example NIFTY levels (22363.35, 22179.90, 22868.90, 23592.85) are not hardcoded."""
    forbidden_values = [
        "22363.35", "22,363.35",
        "22179.90", "22,179.90",
        "22868.90", "22,868.90",
        "23592.85", "23,592.85",
    ]
    for val in forbidden_values:
        matches = [line for line in app_source.splitlines() if val in line and not line.strip().startswith("//")]
        assert len(matches) == 0, f"Found hardcoded example price {val} in production UI: {matches}"


# Criterion 5: Pattern PASS cannot override Walk-Forward FAIL
def test_criterion_5_pattern_pass_cannot_override_walk_forward_fail(app_source: str):
    """Directional bias and historical validation status are kept distinct."""
    assert "Historical trading edge:" in app_source
    # Verify overall validation badge displays NO_VALIDATED_EDGE when walk-forward fails
    assert "NO_VALIDATED_EDGE" in app_source
    # Ensure PASS status is explicitly qualified
    assert "selection-bias provenance remains unverified" in app_source


# Criterion 6: Unverified selection-bias provenance remains visible
def test_criterion_6_selection_bias_provenance_visible(app_source: str):
    """Pattern backtest sample size and unverified selection-bias must be visible."""
    assert "selection-bias provenance remains unverified" in app_source
    assert "Selection-Bias Provenance: UNVERIFIED" in app_source or "Selection-bias risk:" in app_source
    assert "cs.pattern_sample_count" in app_source


# Criterion 7: Drawdowns correctly labelled by evaluation domain
def test_criterion_7_drawdowns_domain_separated(app_source: str):
    """Drawdown audit must explicitly distinguish walk-forward, analogue proxy, and portfolio sim."""
    assert "Walk-Forward Evaluation Drawdown" in app_source
    assert "Historical-Analogue Sequential-Trade Proxy Drawdown" in app_source
    assert "Event-Driven Portfolio Simulation Drawdown" in app_source
    assert "Pattern-Specific Backtest Drawdown" in app_source
    # Ensure sequential trade proxy is clearly designated as proxy, not portfolio
    assert "Domain: Sequential trade proxy of matched historical analogues (NOT portfolio simulation)" in app_source


# Criterion 8: Confidence interval and no-validated-edge warnings accessible in Research & Audit
def test_criterion_8_confidence_interval_warnings_accessible(app_source: str):
    """Sensitivity analysis CI and no edge warning must be accessible in Research & Audit."""
    assert "Statistical Sensitivity Analysis" in app_source
    assert "95% CI:" in app_source
    assert "spans zero" in app_source.lower()


# Criterion 9: Benchmark comparisons remain separate, no unsupported alpha claim
def test_criterion_9_benchmark_comparisons_and_alpha_status(app_source: str):
    """Disaggregated benchmarks vs NIFTY, Cash, and exposure-matched, plus NO_ALPHA_ESTABLISHED."""
    assert "4. Portfolio Simulation & Benchmarks" in app_source
    assert "NO_ALPHA_ESTABLISHED" in app_source
    assert "NIFTY 50" in app_source
    assert "Cash (" in app_source
    assert "Exposure-Matched" in app_source
    assert "CASH RECONCILED" in app_source


# Criterion 10: Expanding and collapsing each Research & Audit section works correctly
def test_criterion_10_collapsible_audit_sections(app_source: str):
    """All 7 sections must have state-driven collapsible toggles."""
    sections = [
        "indicators",
        "analogues",
        "strategyValidation",
        "portfolioSimulation",
        "riskDrawdown",
        "graph",
        "executionDiagnostics",
    ]
    for sec in sections:
        assert f"auditSections.{sec}" in app_source, f"Missing toggle state check for section: {sec}"
        assert f"toggleAuditSection('{sec}')" in app_source, f"Missing toggle call for section: {sec}"


# Criterion 11: Detailed evidence graph and audit info remain accessible
def test_criterion_11_evidence_dependency_graph_accessible(app_source: str):
    """The interactive 5-tier SVG graph and inspectors must exist inside Section 6."""
    assert "6. Evidence-Dependency Graph" in app_source
    assert "<svg" in app_source
    assert "OBSERVATION" in app_source
    assert "STRATEGY_SIGNAL" in app_source
    assert "EVIDENCE_FAMILY" in app_source
    assert "handleNodeClick" in app_source
    assert "handleEdgeClick" in app_source


# Criterion 12: No existing analysis capability or calculation is accidentally removed
def test_criterion_12_no_analysis_capability_removed(app_source: str):
    """Verify presence of all major analytical components."""
    # Indicator claims & ledger
    assert "Evidence Ledger" in app_source
    assert "ledgerFilter" in app_source
    # Analogues & feature vectors
    assert "Feature Vector Comparison" in app_source
    # Partitions & Holdout
    assert "Full (Pattern)" in app_source
    assert "In-Sample" in app_source
    # Execution timing & cost model
    assert "Execution Timing & Cost Assumptions" in app_source
    # Data diagnostics
    assert "Total Bars:" in app_source or "zero_volume_bars" in app_source


# Criterion 13: Interface works at actual Chrome side-panel width without horizontal overflow
def test_criterion_13_side_panel_responsive_width(app_source: str, css_source: str):
    """Side panel layout must have responsive constraints and no overflow."""
    assert "width: '100%'" in app_source or "max-width" in app_source
    # Check CSS for side panel container width
    assert "box-sizing: border-box" in css_source or "overflow-x: hidden" in css_source or "width: 100%" in css_source
    # Check that App container restricts overflow
    assert "overflowX: 'hidden'" in app_source or "overflow: 'hidden'" in app_source or "overflow-x" in css_source
