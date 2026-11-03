"""Simple ALM hedging model for three dated indexed-insurance crediting designs.

Illustrative model. Valuation date 2026-10-26, maturity 2027-10-26 (T = 1).
One S&P 500 point-to-point crediting term per product, closed-form
Black-Scholes with a flat continuous rate, a flat dividend yield, and one
fitted at-the-money-forward volatility. No Monte Carlo, no optimizer, no
policyholder behavior. This module is deliberately self-contained and does not
import from ``src/alm_hedging``.

Products (public product names; terms simplified to one indexed segment):
* Pacific Protective Growth RILA, $100,000: 1y point-to-point on the S&P 500
  (price return, ex dividends), 100% participation, 16.25% cap, 10% buffer,
  no base annual fee.
* Pacific Index Advisory FIA, $100,000: same index and measurement, 100%
  participation, 10% cap, 0% indexed-interest floor, no base annual fee.
* Pacific Horizon IUL 2, 1-Year High Cap Indexed Account, $10,000 indexed
  account value: same index and measurement, 100% participation, 12% cap, 0%
  indexed-interest floor before charges, simplified annual deduction of 0.80
  percentage points of indexed account value.

Accounting definitions
----------------------
Account value at maturity ``AV_T = AV_0 * (1 + credited rate)`` where
``credited rate = participation * option package payoff per $1 - annual deduction``.
The liability is the PV of ``AV_T`` and is split into three components:
``bond`` = PV of returning ``AV_0``; ``option`` = PV of the embedded option
package; ``fee`` = PV of the deterministic deduction. ``liability = bond +
option - fee`` (policyholder view, a positive number the insurer owes).

Insurer sign convention: insurer value = ``-(bond + option) + fee``. Every
P&L in this module is a change in insurer value, so a rise in the liability is
a negative number. Greeks of the liability are reported in dollars per 1%
index move (delta), per 1%^2 (gamma), per volatility point (vega), and per
basis point (rho). Because payoffs depend only on the index return, the index
level is a normalization: every dollar output is invariant to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

OptionKind = str  # "call" or "put"

VALUATION_DATE = date(2026, 10, 26)
MATURITY_DATE = date(2027, 10, 26)
INDEX_NAME = "S&P 500 price return (ex dividends)"


def continuous_rate_from_quoted_yield(quoted_yield: float) -> float:
    """Annually compounded quoted yield to a continuous rate: ``ln(1 + y)``."""
    return math.log1p(quoted_yield)


@dataclass(frozen=True)
class Market:
    """Flat market as of the valuation date.

    ``spot`` is a normalized index level (default 100). ``quoted_yield`` is the
    observed 1y Treasury constant-maturity yield; ``rate`` is its continuous
    equivalent. ``dividend`` is the S&P 500 dividend yield and ``vol`` the
    fitted 1y at-the-money-forward implied volatility applied uniformly.
    """

    spot: float = 100.0
    quoted_yield: float = 0.0447
    dividend: float = 0.01064
    vol: float = 0.1672
    maturity: float = 1.0
    valuation_date: date = VALUATION_DATE
    maturity_date: date = MATURITY_DATE

    @property
    def rate(self) -> float:
        return continuous_rate_from_quoted_yield(self.quoted_yield)

    def forward_factor(self) -> float:
        return math.exp((self.rate - self.dividend) * self.maturity)

    def discount_factor(self) -> float:
        return math.exp(-self.rate * self.maturity)


@dataclass(frozen=True)
class InputSource:
    """One model input with its provenance: observed data or a fitted/model choice."""

    name: str
    value: str
    kind: str  # "observed" or "fitted/model"
    source: str
    url: str
    note: str = ""


def input_sources(market: Market) -> tuple[InputSource, ...]:
    """Inputs, their values, and whether they are observed or fitted/model assumptions."""
    return (
        InputSource(
            "1y Treasury constant-maturity yield",
            f"{market.quoted_yield:.2%} (quoted, {market.valuation_date.isoformat()})",
            "observed",
            "FRED series DGS1 (Board of Governors H.15)",
            "https://fred.stlouisfed.org/series/DGS1",
            "Daily series; the value on the valuation date is used as quoted.",
        ),
        InputSource(
            "Continuous risk-free rate",
            f"{market.rate:.4%} = ln(1 + {market.quoted_yield:.4f})",
            "fitted/model",
            "Derived from DGS1",
            "https://fred.stlouisfed.org/series/DGS1",
            "Treats the quoted yield as annually compounded; bond-equivalent and day-count conventions are ignored.",
        ),
        InputSource(
            "S&P 500 dividend yield",
            f"{market.dividend:.3%} (continuous)",
            "observed",
            "Robert Shiller, Yale, U.S. stock market data (dividends and prices)",
            "http://www.econ.yale.edu/~shiller/data.htm",
            "Trailing dividend yield from the Shiller monthly series around the valuation date; applied as a flat continuous yield.",
        ),
        InputSource(
            "1y at-the-money-forward implied volatility",
            f"{market.vol:.2%} (uniform across strikes)",
            "fitted/model",
            "Fitted from Cboe Option EOD Summary (SPX end-of-day option quotes)",
            "https://datashop.cboe.com/option-eod-summary",
            "A single ATMF vol is fitted to the ~1y expiry and applied to every strike; skew and term structure are ignored.",
        ),
        InputSource(
            "Volatility cross-check",
            "VIX level on the valuation date",
            "observed",
            "Cboe VIX historical data",
            "https://www.cboe.com/tradable_products/vix/vix_historical_data/",
            "30-day index used only as a sanity check on the fitted 1y level, not as a model input.",
        ),
        InputSource(
            "Index level",
            f"{market.spot:.0f} (normalized)",
            "fitted/model",
            "Model convention",
            "",
            "Payoffs depend only on the index return, so dollar results do not depend on the level.",
        ),
    )


EXCLUSIONS: tuple[str, ...] = (
    "Other IUL charges: cost of insurance, premium loads, per-policy and per-unit expense charges, rider charges.",
    "Monthly mechanics: monthly deductions, monthly sweeps into segments, intra-year segment timing.",
    "Insurance coverage: the IUL death benefit and any mortality or lapse assumptions.",
    "Premiums: future premium payments and premium-dependent credits or bonuses.",
    "Withdrawals and surrender: free-withdrawal amounts, surrender charges, and surrender behavior.",
    "RILA early exit and MVA: interim values, market value adjustments, and performance locks before maturity.",
    "Optional benefits: riders, guaranteed living or death benefits, and return-of-premium features.",
    "Loans: IUL policy loans, loan interest, and indexed-loan arbitrage.",
    "Adviser fees: advisory, platform, or third-party fees deducted from account value.",
    "Future years: renewal caps, participation rates, and spreads after the first indexed term.",
    "Real valuation and hedging costs: bid-ask spreads, dealer margins, futures margin and roll costs, collateral, and transaction costs.",
)

CONSIDERED: tuple[str, ...] = (
    "One full indexed term for each product, dated 2026-10-26 to 2027-10-26, with the stated cap, buffer or floor, 100% participation, and the IUL's simplified 0.80-point annual deduction.",
    "Each crediting design decomposed into vanilla S&P 500 option legs per dollar of account value, so the three products price off one model and their risks add.",
    "The liability split into bond, option, and fee components with an explicit insurer sign convention, so interest budget and pricing margin are visible.",
    "Closed-form Black-Scholes pricing with analytic delta, gamma, vega, rho, and theta, checked against finite differences in the test suite.",
    "Observed market inputs (Treasury yield, dividend yield) separated from fitted or model assumptions (continuous-rate conversion, uniform fitted volatility, normalized index level).",
    "Two hedge constructions: exact static option replication and an initial index-futures delta hedge, with residual gamma, vega, and rho reported.",
    "Full revaluation under index, volatility, rate, and combined selloff shocks, before and after the futures hedge, with the bond and option pieces shown separately.",
    "Rebalancing triggers based on the residual index exposure after plausible index, volatility, and time moves.",
)

NEXT_STEPS: tuple[str, ...] = (
    "Replace the uniform volatility with the observed SPX strike skew and term structure, which will raise the value of the RILA's buffer put and lower the cap calls.",
    "Use the full Treasury curve and a dividend futures or forecast curve instead of flat continuous inputs.",
    "Add the excluded IUL mechanics (cost of insurance, monthly deductions, loans) and the RILA interim value and MVA.",
    "Introduce policyholder behavior (lapse, withdrawal, surrender) and multi-year renewals with cap resets.",
    "Price the hedge with real costs: dealer spreads on the option package, futures margin, roll, and financing, and compare the static and dynamic strategies on a cost-adjusted basis.",
    "Extend to Monte Carlo or a volatility-surface model so that gamma and vega hedges (listed options, variance swaps) can be sized alongside the futures, as the full framework in src/alm_hedging does.",
    "Back-test the rebalancing rule through the October 2026 selloff path and report realized hedge slippage.",
)


@dataclass(frozen=True)
class Greeks:
    """Price and sensitivities in natural units.

    delta per index point, gamma per index point squared, vega per 1.00 of
    volatility, rho per 1.00 of rate, theta per year.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    rho: float
    theta: float

    def __add__(self, other: Greeks) -> Greeks:
        return Greeks(
            *(a + b for a, b in zip(self.as_tuple(), other.as_tuple(), strict=True))
        )

    def __sub__(self, other: Greeks) -> Greeks:
        return self + other.scale(-1.0)

    def scale(self, k: float) -> Greeks:
        return Greeks(*(k * a for a in self.as_tuple()))

    def as_tuple(self) -> tuple[float, ...]:
        return (self.price, self.delta, self.gamma, self.vega, self.rho, self.theta)


ZERO_GREEKS = Greeks(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
    kind: OptionKind,
) -> Greeks:
    """Black-Scholes price and analytic Greeks for a European call or put."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    sign = 1.0 if kind == "call" else -1.0
    if maturity <= 0.0 or vol <= 0.0:
        # Expired or deterministic: discounted forward intrinsic value, no convexity.
        fwd = spot * math.exp(-dividend * maturity)
        k_pv = strike * math.exp(-rate * maturity)
        intrinsic = max(sign * (fwd - k_pv), 0.0)
        in_money = sign * (fwd - k_pv) > 0.0
        return Greeks(
            intrinsic,
            sign * math.exp(-dividend * maturity) if in_money else 0.0,
            0.0,
            0.0,
            sign * strike * maturity * math.exp(-rate * maturity) if in_money else 0.0,
            0.0,
        )
    sqrt_t = math.sqrt(maturity)
    disc_q = math.exp(-dividend * maturity)
    disc_r = math.exp(-rate * maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity) / (
        vol * sqrt_t
    )
    d2 = d1 - vol * sqrt_t
    n_d1 = _norm_pdf(d1)
    price = sign * (
        spot * disc_q * _norm_cdf(sign * d1) - strike * disc_r * _norm_cdf(sign * d2)
    )
    delta = sign * disc_q * _norm_cdf(sign * d1)
    gamma = disc_q * n_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * n_d1 * sqrt_t
    rho = sign * strike * maturity * disc_r * _norm_cdf(sign * d2)
    theta = (
        -spot * disc_q * n_d1 * vol / (2.0 * sqrt_t)
        - sign * rate * strike * disc_r * _norm_cdf(sign * d2)
        + sign * dividend * spot * disc_q * _norm_cdf(sign * d1)
    )
    return Greeks(price, delta, gamma, vega, rho, theta)


def bs_call(
    spot: float, strike: float, t: float, r: float, q: float, vol: float
) -> float:
    return bs_greeks(spot, strike, t, r, q, vol, "call").price


def bs_put(
    spot: float, strike: float, t: float, r: float, q: float, vol: float
) -> float:
    return bs_greeks(spot, strike, t, r, q, vol, "put").price


def cash_flow_greeks(amount: float, market: Market) -> Greeks:
    """PV and Greeks of a fixed ``amount`` paid at maturity (the bond and fee pieces)."""
    pv = amount * market.discount_factor()
    return Greeks(pv, 0.0, 0.0, 0.0, -market.maturity * pv, market.rate * pv)


@dataclass(frozen=True)
class Leg:
    """One vanilla option leg per $1 of account value, strike as a multiple of the start index."""

    kind: OptionKind
    moneyness: float
    quantity: float
    label: str

    def payoff_per_dollar(self, index_ratio: float) -> float:
        intrinsic = (
            max(index_ratio - self.moneyness, 0.0)
            if self.kind == "call"
            else max(self.moneyness - index_ratio, 0.0)
        )
        return self.quantity * intrinsic

    def greeks(self, market: Market, start_index: float) -> Greeks:
        """Greeks of ``quantity`` options per index unit, struck at ``moneyness * start_index``."""
        raw = bs_greeks(
            market.spot,
            self.moneyness * start_index,
            market.maturity,
            market.rate,
            market.dividend,
            market.vol,
            self.kind,
        )
        return raw.scale(self.quantity)


@dataclass(frozen=True)
class Product:
    """One indexed crediting segment expressed as option legs plus a fixed deduction."""

    name: str
    product_name: str
    account_value: float
    legs: Sequence[Leg]
    cap: float
    participation: float = 1.0
    floor: float | None = None  # 0.0 for FIA/IUL; None for the RILA buffer design
    buffer: float | None = None  # 0.10 for the RILA; None otherwise
    annual_fee: float = 0.0  # deduction in percentage points of account value
    start_index: float = 100.0
    index: str = INDEX_NAME
    crediting: str = "1-year point-to-point"

    @property
    def units(self) -> float:
        """Option units per leg quantity: one index unit at the start level per start-index dollars."""
        return self.account_value / self.start_index

    def option_payoff_per_dollar(self, index_return: float) -> float:
        ratio = 1.0 + index_return
        return self.participation * sum(
            leg.payoff_per_dollar(ratio) for leg in self.legs
        )

    def credited_rate(self, index_return: float) -> float:
        """Credited rate after the deduction, e.g. 0.12 - 0.008 for the IUL at or above its cap."""
        return self.option_payoff_per_dollar(index_return) - self.annual_fee

    def account_value_at_maturity(self, index_return: float) -> float:
        return self.account_value * (1.0 + self.credited_rate(index_return))

    @property
    def fee_amount(self) -> float:
        return self.annual_fee * self.account_value

    @property
    def terms(self) -> str:
        protection = (
            f"{self.buffer:.0%} buffer"
            if self.buffer is not None
            else f"{self.floor:.0%} floor"
            if self.floor is not None
            else "no protection"
        )
        fee = (
            f"{self.annual_fee * 100:.2f}-pt annual deduction"
            if self.annual_fee
            else "no base annual fee"
        )
        return f"{self.crediting}, {self.participation:.0%} participation, {self.cap:.2%} cap, {protection}, {fee}"


def rila(
    cap: float = 0.1625,
    buffer: float = 0.10,
    account_value: float = 100_000.0,
    start_index: float = 100.0,
) -> Product:
    """Pacific Protective Growth RILA: long ATM call, short call at the cap, short put at the buffer."""
    return Product(
        name="RILA",
        product_name="Pacific Protective Growth RILA",
        account_value=account_value,
        legs=(
            Leg("call", 1.0, 1.0, "Long ATM call (participation)"),
            Leg("call", 1.0 + cap, -1.0, f"Short {1 + cap:.4f} call (cap)"),
            Leg("put", 1.0 - buffer, -1.0, f"Short {1 - buffer:.2f} put (buffer)"),
        ),
        cap=cap,
        buffer=buffer,
        start_index=start_index,
    )


def fia(
    cap: float = 0.10,
    account_value: float = 100_000.0,
    start_index: float = 100.0,
) -> Product:
    """Pacific Index Advisory FIA: 0% floor and a cap, i.e. a call spread."""
    return Product(
        name="FIA",
        product_name="Pacific Index Advisory FIA",
        account_value=account_value,
        legs=(
            Leg("call", 1.0, 1.0, "Long ATM call (participation)"),
            Leg("call", 1.0 + cap, -1.0, f"Short {1 + cap:.2f} call (cap)"),
        ),
        cap=cap,
        floor=0.0,
        start_index=start_index,
    )


def iul(
    cap: float = 0.12,
    annual_fee: float = 0.008,
    account_value: float = 10_000.0,
    start_index: float = 100.0,
) -> Product:
    """Pacific Horizon IUL 2, 1-Year High Cap Indexed Account: call spread less a fixed deduction."""
    return Product(
        name="IUL",
        product_name="Pacific Horizon IUL 2 (1-Year High Cap Indexed Account)",
        account_value=account_value,
        legs=(
            Leg("call", 1.0, 1.0, "Long ATM call (participation)"),
            Leg("call", 1.0 + cap, -1.0, f"Short {1 + cap:.2f} call (cap)"),
        ),
        cap=cap,
        floor=0.0,
        annual_fee=annual_fee,
        start_index=start_index,
    )


def default_products(start_index: float = 100.0) -> tuple[Product, ...]:
    return (
        rila(start_index=start_index),
        fia(start_index=start_index),
        iul(start_index=start_index),
    )


@dataclass(frozen=True)
class LegValuation:
    label: str
    kind: OptionKind
    moneyness: float
    quantity: float
    notional: float  # dollars of account value the leg is written on
    greeks: Greeks  # dollar PV and natural-unit Greeks on the full notional


@dataclass(frozen=True)
class ProductValuation:
    """Liability = bond + option - fee (policyholder view); insurer value is its negative."""

    product: Product
    bond: Greeks
    option: Greeks
    fee: Greeks
    legs: Sequence[LegValuation]

    @property
    def total(self) -> Greeks:
        return self.bond + self.option - self.fee

    @property
    def liability_pv(self) -> float:
        return self.total.price

    @property
    def insurer_value(self) -> float:
        return -self.liability_pv

    @property
    def pricing_margin(self) -> float:
        """Premium received less the liability PV at inception."""
        return self.product.account_value - self.liability_pv

    @property
    def interest_budget(self) -> float:
        """What the premium earns at the risk-free rate over the term: AV_0 - bond PV."""
        return self.product.account_value - self.bond.price


def price_product(product: Product, market: Market) -> ProductValuation:
    leg_vals = []
    option = ZERO_GREEKS
    for leg in product.legs:
        g = leg.greeks(market, product.start_index).scale(
            product.units * product.participation
        )
        leg_vals.append(
            LegValuation(
                leg.label,
                leg.kind,
                leg.moneyness,
                leg.quantity * product.participation,
                product.account_value * abs(leg.quantity) * product.participation,
                g,
            )
        )
        option = option + g
    return ProductValuation(
        product=product,
        bond=cash_flow_greeks(product.account_value, market),
        option=option,
        fee=cash_flow_greeks(product.fee_amount, market),
        legs=tuple(leg_vals),
    )


def dollar_greeks(g: Greeks, market: Market) -> dict[str, float]:
    """Dollar sensitivities: per 1% index move, per 1%^2, per vol point, per basis point."""
    s = market.spot
    return {
        "PV ($)": g.price,
        "Delta ($ per 1% index)": g.delta * s / 100.0,
        "Gamma ($ per 1%^2)": g.gamma * s * s / 10_000.0,
        "Vega ($ per vol pt)": g.vega * 0.01,
        "Rho ($ per bp)": g.rho * 1e-4,
    }


def solve_cap_from_budget(
    builder: Callable[[float], Product],
    budget: float,
    market: Market,
    lo: float = 0.0005,
    hi: float = 2.0,
    tol: float = 1e-10,
) -> float:
    """Cap whose option package PV equals ``budget`` dollars, by bisection (PV rises with the cap)."""
    f_lo = price_product(builder(lo), market).option.price - budget
    f_hi = price_product(builder(hi), market).option.price - budget
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError(
            f"budget {budget:.4f} is outside the attainable PV range [{f_lo + budget:.4f}, {f_hi + budget:.4f}]"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = price_product(builder(mid), market).option.price - budget
        if abs(f_mid) < tol or hi - lo < tol:
            return mid
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Scenario:
    name: str
    index_shock: float = 0.0  # relative, e.g. -0.10
    vol_shock: float = 0.0  # absolute, e.g. 0.05
    rate_shock: float = 0.0  # absolute shift of the continuous rate, e.g. 0.005
    time_elapsed: float = 0.0  # years, for rebalancing checks

    def apply(self, market: Market) -> Market:
        shocked_rate = market.rate + self.rate_shock
        return replace(
            market,
            spot=market.spot * (1.0 + self.index_shock),
            vol=market.vol + self.vol_shock,
            quoted_yield=math.expm1(shocked_rate),
            maturity=market.maturity - self.time_elapsed,
        )


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Index -20%", index_shock=-0.20),
    Scenario("Index -10%", index_shock=-0.10),
    Scenario("Index +10%", index_shock=0.10),
    Scenario("Index +20%", index_shock=0.20),
    Scenario("Vol +5 pts", vol_shock=0.05),
    Scenario("Vol -5 pts", vol_shock=-0.05),
    Scenario("Rates +50 bp", rate_shock=0.005),
    Scenario("Rates -50 bp", rate_shock=-0.005),
    Scenario("Selloff: index -15%, vol +8, rates -25 bp", -0.15, 0.08, -0.0025),
)

REBALANCE_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Index -10%", index_shock=-0.10),
    Scenario("Index -5%", index_shock=-0.05),
    Scenario("Index +5%", index_shock=0.05),
    Scenario("Index +10%", index_shock=0.10),
    Scenario("Vol +5 pts", vol_shock=0.05),
    Scenario("Vol -5 pts", vol_shock=-0.05),
    Scenario("One month elapsed, index unchanged", time_elapsed=1.0 / 12.0),
)


@dataclass(frozen=True)
class ReplicationLeg:
    """Aggregated static-replication position across products."""

    kind: OptionKind
    moneyness: float
    net_notional: float  # signed dollars of account value (long positive)
    cost: float  # dollar PV the insurer would pay (positive) or receive (negative)
    products: str


def static_replication(
    valuations: Sequence[ProductValuation],
) -> list[ReplicationLeg]:
    """Net option positions the insurer would buy to replicate the option component exactly."""
    buckets: dict[tuple[OptionKind, float], list[LegValuation]] = {}
    names: dict[tuple[OptionKind, float], list[str]] = {}
    for val in valuations:
        for leg in val.legs:
            key = (leg.kind, round(leg.moneyness, 6))
            buckets.setdefault(key, []).append(leg)
            names.setdefault(key, []).append(val.product.name)
    legs = []
    for key, items in buckets.items():
        net_notional = sum(math.copysign(leg.notional, leg.quantity) for leg in items)
        cost = sum(leg.greeks.price for leg in items)
        legs.append(
            ReplicationLeg(key[0], key[1], net_notional, cost, "+".join(names[key]))
        )
    return sorted(legs, key=lambda leg: (leg.kind, leg.moneyness))


@dataclass(frozen=True)
class Hedge:
    """Long index futures sized to the liability delta. Units are in normalized index units."""

    futures_units: float
    futures_notional: float  # dollars of index exposure = liability dollar delta
    liability_delta: float  # natural units, per index point
    residual_gamma: float  # $ per 1%^2, insurer view
    residual_vega: float  # $ per vol pt, insurer view
    residual_rho: float  # $ per bp, insurer view
    option_rho: float  # $ per bp of the option component alone, insurer view

    def pnl(self, base: Market, shocked: Market) -> float:
        forward_base = base.spot * base.forward_factor()
        forward_shocked = shocked.spot * shocked.forward_factor()
        return self.futures_units * (forward_shocked - forward_base)

    def dollar_delta(self, market: Market) -> float:
        """Dollar delta of the futures per 1% index move at ``market``."""
        return self.futures_units * market.forward_factor() * market.spot / 100.0


def portfolio_total(valuations: Sequence[ProductValuation]) -> Greeks:
    total = ZERO_GREEKS
    for val in valuations:
        total = total + val.total
    return total


def portfolio_option(valuations: Sequence[ProductValuation]) -> Greeks:
    total = ZERO_GREEKS
    for val in valuations:
        total = total + val.option
    return total


def delta_hedge(valuations: Sequence[ProductValuation], market: Market) -> Hedge:
    """Buy index futures against the liability delta; report what the linear hedge leaves.

    The insurer is short the liability, so its delta is minus the liability
    delta. A futures unit has delta ``forward_factor`` per index unit, so
    ``delta / forward_factor`` units offset it. Futures carry no gamma or vega;
    their small rho (through the forward) is included in the stress P&L.
    """
    total = portfolio_total(valuations)
    option = portfolio_option(valuations)
    g_total = dollar_greeks(total, market)
    g_option = dollar_greeks(option, market)
    units = total.delta / market.forward_factor()
    return Hedge(
        futures_units=units,
        futures_notional=total.delta * market.spot,
        liability_delta=total.delta,
        residual_gamma=-g_total["Gamma ($ per 1%^2)"],
        residual_vega=-g_total["Vega ($ per vol pt)"],
        residual_rho=-g_total["Rho ($ per bp)"],
        option_rho=-g_option["Rho ($ per bp)"],
    )


@dataclass(frozen=True)
class StressRow:
    """Insurer-view P&L by component under one full-revaluation scenario."""

    scenario: str
    liability_pv: float
    bond_pnl: float
    option_pnl: float
    fee_pnl: float
    hedge_pnl: float

    @property
    def unhedged_pnl(self) -> float:
        return self.bond_pnl + self.option_pnl + self.fee_pnl

    @property
    def hedged_pnl(self) -> float:
        return self.unhedged_pnl + self.hedge_pnl

    @property
    def option_hedged_pnl(self) -> float:
        """Option component plus futures: the part a derivatives desk is asked to hedge."""
        return self.option_pnl + self.hedge_pnl


def stress_table(
    products: Sequence[Product],
    market: Market,
    hedge: Hedge | None = None,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
) -> list[StressRow]:
    """Full revaluation of every component under each scenario, with and without the futures."""
    base = [price_product(p, market) for p in products]
    rows = []
    for scenario in scenarios:
        shocked_market = scenario.apply(market)
        shocked = [price_product(p, shocked_market) for p in products]
        bond = -sum(
            s.bond.price - b.bond.price for s, b in zip(shocked, base, strict=True)
        )
        option = -sum(
            s.option.price - b.option.price for s, b in zip(shocked, base, strict=True)
        )
        fee = sum(s.fee.price - b.fee.price for s, b in zip(shocked, base, strict=True))
        hedge_pnl = hedge.pnl(market, shocked_market) if hedge is not None else 0.0
        rows.append(
            StressRow(
                scenario.name,
                sum(s.liability_pv for s in shocked),
                bond,
                option,
                fee,
                hedge_pnl,
            )
        )
    return rows


def pnl_range(values: Sequence[float]) -> float:
    return max(values) - min(values)


@dataclass(frozen=True)
class RebalanceRow:
    """Residual dollar delta of the hedged book after a move, used to set rebalancing triggers."""

    scenario: str
    liability_dollar_delta: float  # $ per 1% index, liability view
    futures_dollar_delta: float  # $ per 1% index
    residual_dollar_delta: float  # insurer view: futures minus liability
    residual_notional: float  # residual_dollar_delta * 100, in index-exposure dollars


def rebalancing_table(
    products: Sequence[Product],
    market: Market,
    hedge: Hedge,
    scenarios: Sequence[Scenario] = REBALANCE_SCENARIOS,
) -> list[RebalanceRow]:
    rows = []
    for scenario in scenarios:
        shocked = scenario.apply(market)
        total = portfolio_total([price_product(p, shocked) for p in products])
        liability = total.delta * shocked.spot / 100.0
        futures = hedge.dollar_delta(shocked)
        residual = futures - liability
        rows.append(
            RebalanceRow(scenario.name, liability, futures, residual, residual * 100.0)
        )
    return rows


@dataclass(frozen=True)
class SimpleModelResults:
    market: Market
    products: Sequence[Product]
    valuations: Sequence[ProductValuation]
    total: Greeks
    option: Greeks
    replication: Sequence[ReplicationLeg]
    hedge: Hedge
    stress: Sequence[StressRow]
    rebalancing: Sequence[RebalanceRow]
    affordable_caps: dict[str, float]
    rebalance_threshold: float
    sources: Sequence[InputSource] = field(default_factory=tuple)
    exclusions: Sequence[str] = EXCLUSIONS
    considered: Sequence[str] = CONSIDERED
    next_steps: Sequence[str] = NEXT_STEPS
    payoff_grid: list[float] = field(default_factory=list)
    credited_rates: dict[str, list[float]] = field(default_factory=dict)

    @property
    def total_account_value(self) -> float:
        return sum(p.account_value for p in self.products)

    @property
    def unhedged_range(self) -> float:
        return pnl_range([r.unhedged_pnl for r in self.stress])

    @property
    def hedged_range(self) -> float:
        return pnl_range([r.hedged_pnl for r in self.stress])

    @property
    def option_unhedged_range(self) -> float:
        return pnl_range([r.option_pnl for r in self.stress])

    @property
    def option_hedged_range(self) -> float:
        return pnl_range([r.option_hedged_pnl for r in self.stress])

    @property
    def static_replication_cost(self) -> float:
        return sum(leg.cost for leg in self.replication)


def run_simple_model(
    market: Market = Market(),
    rebalance_threshold_pct: float = 0.02,
) -> SimpleModelResults:
    """Run every calculation once so every output reconciles to the same numbers.

    ``rebalance_threshold_pct`` is the residual index exposure, as a fraction
    of total account value, beyond which the futures are rebalanced.
    """
    products = default_products(start_index=market.spot)
    valuations = tuple(price_product(p, market) for p in products)
    builders: dict[str, Callable[[float], Product]] = {
        "RILA": lambda c: rila(cap=c, start_index=market.spot),
        "FIA": lambda c: fia(cap=c, start_index=market.spot),
        "IUL": lambda c: iul(cap=c, start_index=market.spot),
    }
    # The cap each design could afford if its whole interest budget (plus any fee) bought options.
    affordable = {
        v.product.name: solve_cap_from_budget(
            builders[v.product.name], v.interest_budget + v.fee.price, market
        )
        for v in valuations
    }
    hedge = delta_hedge(valuations, market)
    grid = [round(0.60 + 0.01 * i, 2) for i in range(81)]
    credited = {p.name: [p.credited_rate(x - 1.0) for x in grid] for p in products}
    return SimpleModelResults(
        market=market,
        products=products,
        valuations=valuations,
        total=portfolio_total(valuations),
        option=portfolio_option(valuations),
        replication=static_replication(valuations),
        hedge=hedge,
        stress=stress_table(products, market, hedge),
        rebalancing=rebalancing_table(products, market, hedge),
        affordable_caps=affordable,
        rebalance_threshold=rebalance_threshold_pct
        * sum(p.account_value for p in products),
        sources=input_sources(market),
        payoff_grid=grid,
        credited_rates=credited,
    )


def recommendation(results: SimpleModelResults) -> str:
    """Hedge recommendation: instruments, sizes, costs, residual risks, rebalancing triggers."""
    h, s, m = results.hedge, results.stress, results.market
    vol_up = next(r for r in s if r.scenario == "Vol +5 pts")
    selloff = next(r for r in s if r.scenario.startswith("Selloff"))
    worst = min(s, key=lambda r: r.option_hedged_pnl)
    gamma_side = "long" if h.residual_gamma > 0 else "short"
    vega_side = "long" if h.residual_vega > 0 else "short"
    breaching = [
        r.scenario
        for r in results.rebalancing
        if abs(r.residual_notional) > results.rebalance_threshold
    ]
    holding = [
        r.scenario
        for r in results.rebalancing
        if abs(r.residual_notional) <= results.rebalance_threshold
    ]
    breach_text = (
        "none of the checked moves breaches it on its own"
        if not breaching
        else "it is breached by " + ", ".join(breaching)
    ) + (f" and not by {', '.join(holding)}" if holding and breaching else "")
    replication = "; ".join(
        f"{'buy' if leg.net_notional > 0 else 'sell'} {leg.kind}s struck at {leg.moneyness:.2%} of the start index on "
        f"${abs(leg.net_notional):,.0f} ({'cost' if leg.cost > 0 else 'premium received'} ${abs(leg.cost):,.0f})"
        for leg in results.replication
    )
    return (
        f"Instruments and sizes. Initial hedge as of {m.valuation_date.isoformat()}: buy S&P 500 index futures with "
        f"${h.futures_notional:,.0f} of index exposure ({h.futures_units:.4f} units at a normalized level of {m.spot:.0f}; "
        f"convert to contracts with the live index level and contract multiplier) against ${results.total_account_value:,.0f} "
        f"of account value, offsetting a liability delta of ${h.liability_delta * m.spot / 100:,.0f} per 1% index move. "
        f"The static alternative replicates the option component exactly: {replication}; net premium "
        f"${results.static_replication_cost:,.0f}, equal to the option PV. "
        f"Costs. The futures require no premium (margin, roll, and financing are excluded); the static replication costs "
        f"${results.static_replication_cost:,.0f} up front, funded by the ${sum(v.interest_budget for v in results.valuations):,.0f} "
        f"interest budget on the bond component, leaving a pricing margin of ${sum(v.pricing_margin for v in results.valuations):,.0f} "
        f"before any real dealer spread. "
        f"Residual risks after the futures hedge. Full revaluation shrinks the option-component P&L range from "
        f"${results.option_unhedged_range:,.0f} to ${results.option_hedged_range:,.0f} across the scenarios. The insurer is "
        f"{gamma_side} gamma (${h.residual_gamma:+,.0f} per 1%^2) and {vega_side} vega (${h.residual_vega:+,.0f} per vol point; "
        f"a +5 pt vol shock moves the hedged book by ${vol_up.option_hedged_pnl:+,.0f}); the worst hedged option-component "
        f"scenario is {worst.scenario} at ${worst.option_hedged_pnl:+,.0f}, and the combined selloff scenario gives "
        f"${selloff.option_hedged_pnl:+,.0f} on the option component and ${selloff.hedged_pnl:+,.0f} in total. Rate risk is "
        f"dominated by the bond component (${h.residual_rho:+,.0f} per bp in total versus ${h.option_rho:+,.0f} per bp on the "
        f"options), which belongs with general-account asset duration rather than the equity hedge. "
        f"Rebalancing triggers. Re-hedge when the residual index exposure exceeds ${results.rebalance_threshold:,.0f} "
        f"({results.rebalance_threshold / results.total_account_value:.0%} of account value); {breach_text}. Also re-hedge after "
        f"a volatility move of 5 points or more, at least monthly, and in the final month before the {m.maturity_date.isoformat()} "
        f"maturity when the caps and buffer pin delta. Illustrative model; see the exclusions for what is not covered."
    )


def explanation(results: SimpleModelResults) -> list[str]:
    """Plain-language reading of the results, one paragraph per theme."""
    m = results.market
    by_name = {v.product.name: v for v in results.valuations}
    rila_v, fia_v, iul_v = by_name["RILA"], by_name["FIA"], by_name["IUL"]
    rila_put = next(leg for leg in rila_v.legs if leg.kind == "put")
    h = results.hedge
    down20 = next(r for r in results.stress if r.scenario == "Index -20%")
    up20 = next(r for r in results.stress if r.scenario == "Index +20%")
    vol_up = next(r for r in results.stress if r.scenario == "Vol +5 pts")
    rates_up = next(r for r in results.stress if r.scenario == "Rates +50 bp")
    selloff = next(r for r in results.stress if r.scenario.startswith("Selloff"))
    budget = sum(v.interest_budget for v in results.valuations)
    return [
        (
            f"Why the RILA cap is higher but its option package is cheaper. The RILA's {rila_v.product.cap:.2%} cap costs "
            f"${rila_v.option.price:,.0f} against ${fia_v.option.price:,.0f} for the FIA's {fia_v.product.cap:.0%} cap because the "
            f"policyholder has written the insurer a put struck at 90% of the start index, worth ${-rila_put.greeks.price:,.0f} on "
            f"$100,000. That premium funds a cap call struck further out of the money. The FIA and IUL are plain call spreads, "
            f"so their cost is set entirely by how far out the cap call sits."
        ),
        (
            f"Where the money comes from. Each premium dollar earns the risk-free rate for one year; on ${results.total_account_value:,.0f} "
            f"that interest budget is ${budget:,.0f}. The option packages cost ${results.option.price:,.0f} and the IUL deduction returns "
            f"${iul_v.fee.price:,.0f}, leaving ${sum(v.pricing_margin for v in results.valuations):,.0f} of pricing margin before expenses, "
            f"dealer spreads, and any credit spread earned above Treasuries. Caps are therefore tightly linked to rates and volatility: "
            f"with the full budget the RILA could afford a {results.affordable_caps['RILA']:.2%} cap, the FIA {results.affordable_caps['FIA']:.2%}, "
            f"and the IUL {results.affordable_caps['IUL']:.2%}."
        ),
        (
            f"What the Greeks say. The liability gains ${h.liability_delta * m.spot / 100:,.0f} for every 1% rise in the index, almost "
            f"all of it from the long at-the-money calls. Gamma and vega are negative for the liability (${-h.residual_gamma:,.0f} per 1%^2 "
            f"and ${-h.residual_vega:,.0f} per vol point), because the short cap calls and the RILA buffer put give back more convexity "
            f"than the at-the-money calls provide at this volatility. Rho on the options is small; the bond component's rho "
            f"(about ${h.residual_rho - h.option_rho:,.0f} per bp from the insurer's side) dominates and is an asset-duration question."
        ),
        (
            f"What the futures hedge does and does not do. Buying ${h.futures_notional:,.0f} of index exposure removes the first-order move: "
            f"an index fall of 20% that would have gained the insurer ${down20.option_pnl:,.0f} on the options nets to ${down20.option_hedged_pnl:+,.0f} "
            f"with the futures, and a 20% rise that would have cost ${-up20.option_pnl:,.0f} nets to ${up20.option_hedged_pnl:+,.0f}. Both residuals "
            f"are positive because the insurer is net long gamma, so large moves help the hedged book. The hedge does nothing for volatility: "
            f"a +5 point vol shock still moves the book by ${vol_up.option_hedged_pnl:+,.0f}, and a vol fall costs money. In the combined "
            f"selloff the option book finishes at ${selloff.option_hedged_pnl:+,.0f} and the total at ${selloff.hedged_pnl:+,.0f} once the bond "
            f"component's rate {'gain' if selloff.bond_pnl > 0 else 'loss'} of ${abs(selloff.bond_pnl):,.0f} is included."
        ),
        (
            f"Rates. A 50 bp rise lowers the PV of the promised account value by ${rates_up.bond_pnl:,.0f} (a gain for the insurer) and "
            f"nudges the options and the futures forward. The equity hedge should not be sized to this; it belongs with the general-account "
            f"assets backing the ${sum(v.bond.price for v in results.valuations):,.0f} bond component."
        ),
        (
            "Limits of this reading. A single fitted volatility understates the buffer put and overstates the cap calls relative to the "
            "observed SPX skew, so the RILA margin is likely optimistic. One year of one segment with no behavior, charges, or hedge "
            "costs is a lower bound on complexity, not a reserve or a price."
        ),
    ]
