"""Tests for the self-contained simple model in ``simple/simple_model.py``."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simple"))

import simple_model as sm  # noqa: E402

MARKET = sm.Market()
DF = math.exp(-MARKET.rate)


def test_dates_and_market_inputs_match_the_specification() -> None:
    assert MARKET.valuation_date == date(2026, 10, 26)
    assert MARKET.maturity_date == date(2027, 10, 26)
    assert MARKET.maturity == 1.0
    assert MARKET.quoted_yield == 0.0447
    assert MARKET.rate == pytest.approx(math.log(1.0447))
    assert MARKET.rate == pytest.approx(0.043730, abs=5e-7)
    assert MARKET.dividend == 0.01064
    assert MARKET.vol == 0.1672


def test_product_terms_and_amounts() -> None:
    rila, fia, iul = sm.default_products()
    assert rila.product_name == "Pacific Protective Growth RILA"
    assert (rila.account_value, rila.cap, rila.buffer, rila.floor) == (
        100_000.0,
        0.1625,
        0.10,
        None,
    )
    assert rila.annual_fee == 0.0
    assert [(leg.kind, leg.moneyness, leg.quantity) for leg in rila.legs] == [
        ("call", 1.0, 1.0),
        ("call", 1.1625, -1.0),
        ("put", 0.90, -1.0),
    ]
    assert fia.product_name == "Pacific Index Advisory FIA"
    assert (fia.account_value, fia.cap, fia.floor, fia.annual_fee) == (
        100_000.0,
        0.10,
        0.0,
        0.0,
    )
    assert [(leg.kind, leg.moneyness) for leg in fia.legs] == [
        ("call", 1.0),
        ("call", 1.10),
    ]
    assert iul.product_name.startswith("Pacific Horizon IUL 2")
    assert (iul.account_value, iul.cap, iul.floor, iul.annual_fee) == (
        10_000.0,
        0.12,
        0.0,
        0.008,
    )
    assert iul.fee_amount == pytest.approx(80.0)
    for product in (rila, fia, iul):
        assert product.participation == 1.0
        assert product.index == sm.INDEX_NAME
        assert product.crediting == "1-year point-to-point"


def test_put_call_parity() -> None:
    spot, strike, t, r, q, vol = 100.0, 105.0, 1.0, MARKET.rate, 0.01064, 0.1672
    call = sm.bs_call(spot, strike, t, r, q, vol)
    put = sm.bs_put(spot, strike, t, r, q, vol)
    expected = spot * math.exp(-q * t) - strike * math.exp(-r * t)
    assert call - put == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("index_return", "expected"),
    [
        (0.25, 0.1625),  # above the cap
        (0.1625, 0.1625),  # exactly at the cap
        (0.12, 0.12),  # between zero and the cap
        (0.0, 0.0),
        (-0.05, 0.0),  # inside the 10% buffer
        (-0.10, 0.0),  # at the buffer edge
        (-0.25, -0.15),  # beyond the buffer
    ],
)
def test_rila_credited_rate_edge_cases(index_return: float, expected: float) -> None:
    assert sm.rila().credited_rate(index_return) == pytest.approx(expected, abs=1e-12)


def test_fia_and_iul_credited_rates_and_account_values() -> None:
    fia, iul = sm.fia(), sm.iul()
    assert fia.credited_rate(-0.30) == 0.0
    assert fia.credited_rate(0.05) == pytest.approx(0.05)
    assert fia.credited_rate(0.30) == pytest.approx(0.10)
    assert fia.account_value_at_maturity(0.30) == pytest.approx(110_000.0)
    # IUL: 0% floor before charges, then the 0.80-point deduction.
    assert iul.credited_rate(-0.30) == pytest.approx(-0.008)
    assert iul.credited_rate(0.05) == pytest.approx(0.042)
    assert iul.credited_rate(0.30) == pytest.approx(0.112)
    assert iul.account_value_at_maturity(0.30) == pytest.approx(11_120.0)
    assert iul.account_value_at_maturity(-0.30) == pytest.approx(9_920.0)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0, 112.0, 116.25])
def test_analytic_greeks_match_finite_differences(kind: str, strike: float) -> None:
    s, t, r, q, vol = 100.0, 1.0, MARKET.rate, MARKET.dividend, MARKET.vol

    def price(s_: float = s, t_: float = t, r_: float = r, vol_: float = vol) -> float:
        return sm.bs_greeks(s_, strike, t_, r_, q, vol_, kind).price

    g = sm.bs_greeks(s, strike, t, r, q, vol, kind)
    h = 1e-3
    assert g.delta == pytest.approx(
        (price(s_=s + h) - price(s_=s - h)) / (2 * h), rel=1e-6
    )
    assert g.gamma == pytest.approx(
        (price(s_=s + h) - 2 * price() + price(s_=s - h)) / (h * h), rel=1e-4
    )
    assert g.vega == pytest.approx(
        (price(vol_=vol + 1e-5) - price(vol_=vol - 1e-5)) / 2e-5, rel=1e-6
    )
    assert g.rho == pytest.approx(
        (price(r_=r + 1e-5) - price(r_=r - 1e-5)) / 2e-5, rel=1e-6
    )
    assert g.theta == pytest.approx(
        -(price(t_=t + 1e-5) - price(t_=t - 1e-5)) / 2e-5, rel=1e-6
    )


def test_bond_option_fee_decomposition_and_reconciliation() -> None:
    for product in sm.default_products():
        val = sm.price_product(product, MARKET)
        assert val.bond.price == pytest.approx(product.account_value * DF)
        assert val.fee.price == pytest.approx(product.fee_amount * DF)
        assert val.option.price == pytest.approx(
            sum(leg.greeks.price for leg in val.legs)
        )
        assert val.liability_pv == pytest.approx(
            val.bond.price + val.option.price - val.fee.price
        )
        assert val.insurer_value == pytest.approx(-val.liability_pv)
        assert val.pricing_margin == pytest.approx(
            product.account_value - val.liability_pv
        )
        assert val.interest_budget == pytest.approx(product.account_value * (1.0 - DF))
        # Fixed cash flows carry rho only; the option package carries everything else.
        assert val.bond.rho == pytest.approx(-MARKET.maturity * val.bond.price)
        assert val.total.rho == pytest.approx(
            val.option.rho + val.bond.rho - val.fee.rho
        )
        assert (val.bond.delta, val.bond.gamma, val.bond.vega) == (0.0, 0.0, 0.0)
        assert val.total.delta == pytest.approx(val.option.delta)


def test_iul_fee_is_valued_and_reduces_the_liability() -> None:
    val = sm.price_product(sm.iul(), MARKET)
    assert val.fee.price == pytest.approx(80.0 * DF)
    without_fee = sm.price_product(sm.iul(annual_fee=0.0), MARKET)
    assert without_fee.liability_pv - val.liability_pv == pytest.approx(80.0 * DF)
    assert without_fee.option.price == pytest.approx(val.option.price)


def test_product_dollar_greeks_match_full_revaluation() -> None:
    product = sm.rila()

    def liability(market: sm.Market) -> float:
        return sm.price_product(product, market).liability_pv

    g = sm.dollar_greeks(sm.price_product(product, MARKET).total, MARKET)
    up, dn = replace(MARKET, spot=100.001), replace(MARKET, spot=99.999)
    assert g["Delta ($ per 1% index)"] == pytest.approx(
        (liability(up) - liability(dn)) / 0.002, rel=1e-6
    )
    assert g["Gamma ($ per 1%^2)"] == pytest.approx(
        (liability(up) - 2 * liability(MARKET) + liability(dn)) / 0.001**2, rel=1e-4
    )
    v_up, v_dn = (
        replace(MARKET, vol=MARKET.vol + 1e-5),
        replace(MARKET, vol=MARKET.vol - 1e-5),
    )
    assert g["Vega ($ per vol pt)"] == pytest.approx(
        (liability(v_up) - liability(v_dn)) / 2e-5 * 0.01, rel=1e-6
    )
    r_up = replace(MARKET, quoted_yield=math.expm1(MARKET.rate + 1e-6))
    r_dn = replace(MARKET, quoted_yield=math.expm1(MARKET.rate - 1e-6))
    assert g["Rho ($ per bp)"] == pytest.approx(
        (liability(r_up) - liability(r_dn)) / 2e-6 * 1e-4, rel=1e-5
    )


def test_static_replication_nets_positions_and_reconciles_to_option_pv() -> None:
    results = sm.run_simple_model()
    legs = {(leg.kind, leg.moneyness): leg for leg in results.replication}
    assert legs[("call", 1.0)].net_notional == pytest.approx(210_000.0)
    assert legs[("call", 1.10)].net_notional == pytest.approx(-100_000.0)
    assert legs[("call", 1.12)].net_notional == pytest.approx(-10_000.0)
    assert legs[("call", 1.1625)].net_notional == pytest.approx(-100_000.0)
    assert legs[("put", 0.90)].net_notional == pytest.approx(-100_000.0)
    assert legs[("call", 1.0)].products == "RILA+FIA+IUL"
    assert results.static_replication_cost == pytest.approx(results.option.price)
    assert results.option.price == pytest.approx(
        sum(v.option.price for v in results.valuations)
    )


def test_cap_solver_round_trip() -> None:
    for builder, cap in (
        (sm.fia, 0.10),
        (sm.iul, 0.12),
        (lambda c: sm.rila(cap=c), 0.1625),
    ):
        budget = sm.price_product(builder(cap), MARKET).option.price
        solved = sm.solve_cap_from_budget(builder, budget, MARKET)
        assert solved == pytest.approx(cap, abs=1e-7)
    with pytest.raises(ValueError):
        sm.solve_cap_from_budget(sm.fia, 1e9, MARKET)


def test_delta_hedge_offsets_dollar_delta_and_reports_residuals() -> None:
    results = sm.run_simple_model()
    h = results.hedge
    total = sm.dollar_greeks(results.total, MARKET)
    assert h.futures_units > 0.0
    assert h.dollar_delta(MARKET) == pytest.approx(total["Delta ($ per 1% index)"])
    assert h.futures_notional == pytest.approx(results.total.delta * MARKET.spot)
    assert h.residual_gamma == pytest.approx(-total["Gamma ($ per 1%^2)"])
    assert h.residual_vega == pytest.approx(-total["Vega ($ per vol pt)"])
    assert h.residual_rho == pytest.approx(-total["Rho ($ per bp)"])
    base = sm.rebalancing_table(
        results.products, MARKET, h, scenarios=(sm.Scenario("base"),)
    )
    assert base[0].residual_dollar_delta == pytest.approx(0.0, abs=1e-9)


def test_stress_components_sum_and_hedge_reduces_option_range() -> None:
    results = sm.run_simple_model()
    for row in results.stress:
        assert row.unhedged_pnl == pytest.approx(
            row.bond_pnl + row.option_pnl + row.fee_pnl
        )
        assert row.hedged_pnl == pytest.approx(row.unhedged_pnl + row.hedge_pnl)
        if row.scenario.startswith(("Index", "Vol")):
            assert row.bond_pnl == pytest.approx(0.0, abs=1e-9)
            assert row.fee_pnl == pytest.approx(0.0, abs=1e-9)
        if row.scenario.startswith("Index"):
            assert abs(row.hedged_pnl) < abs(row.unhedged_pnl)
    rates_up = next(r for r in results.stress if r.scenario == "Rates +50 bp")
    bond_pv = sum(v.bond.price for v in results.valuations)
    assert rates_up.bond_pnl == pytest.approx(bond_pv * (1.0 - math.exp(-0.005)))
    assert results.option_hedged_range < 0.25 * results.option_unhedged_range
    assert results.hedged_range < results.unhedged_range
    unhedged = sm.stress_table(results.products, MARKET)
    assert all(row.hedge_pnl == 0.0 for row in unhedged)


def test_dollar_results_do_not_depend_on_the_index_level() -> None:
    base = sm.run_simple_model()
    rescaled = sm.run_simple_model(sm.Market(spot=6_800.0))
    assert rescaled.total.price == pytest.approx(base.total.price)
    assert rescaled.hedge.futures_notional == pytest.approx(base.hedge.futures_notional)
    assert sm.dollar_greeks(rescaled.total, rescaled.market) == pytest.approx(
        sm.dollar_greeks(base.total, base.market)
    )
    for a, b in zip(rescaled.stress, base.stress, strict=True):
        assert a.hedged_pnl == pytest.approx(b.hedged_pnl)


def test_sources_exclusions_and_recommendation() -> None:
    results = sm.run_simple_model()
    urls = {s.url for s in results.sources}
    assert "https://fred.stlouisfed.org/series/DGS1" in urls
    assert "http://www.econ.yale.edu/~shiller/data.htm" in urls
    assert "https://datashop.cboe.com/option-eod-summary" in urls
    assert "https://www.cboe.com/tradable_products/vix/vix_historical_data/" in urls
    kinds = {s.name: s.kind for s in results.sources}
    assert kinds["1y Treasury constant-maturity yield"] == "observed"
    assert kinds["1y at-the-money-forward implied volatility"] == "fitted/model"
    joined = " ".join(results.exclusions)
    for term in (
        "IUL charges",
        "Monthly",
        "Insurance coverage",
        "Premiums",
        "surrender",
        "MVA",
        "Optional benefits",
        "Loans",
        "Adviser fees",
        "Future years",
        "hedging costs",
    ):
        assert term in joined
    text = sm.recommendation(results)
    for term in (
        "futures",
        "Costs.",
        "Residual risks",
        "Rebalancing triggers",
        "2027-10-26",
    ):
        assert term in text


def test_report_builds_from_the_shared_results(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import build_report

    results = sm.run_simple_model()
    assert len(sm.explanation(results)) >= 5
    assert len(results.considered) >= 5 and len(results.next_steps) >= 5
    out = build_report.build_report(results, tmp_path / "report.pdf")
    assert out.exists() and out.stat().st_size > 20_000
    assert out.read_bytes()[:5] == b"%PDF-"
    one = build_report.build_one_page(results, tmp_path / "one-page.pdf")
    assert one.exists() and one.read_bytes()[:5] == b"%PDF-"
    data = one.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 1
