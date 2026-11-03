"""Render the simple-model report to ``simple/simple-model-report.pdf``.

Usage::

    python simple/build_report.py [--output PATH]

Structure: 1 product terms; 2 cash flows and embedded options; 3 valuation and
Greeks; 4 net exposures; 5 hedge recommendations; 6 explanation of the
results; then data sources, what was considered, what was not, and next steps.
Every number comes from one ``run_simple_model()`` call, the same function the
notebook uses. Illustrative model.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simple_model as sm  # noqa: E402

INK = "#14213D"
SLATE = "#68778D"
PALE = "#EAF0F8"
RULE = "#CBD5E1"
COLORS = {"RILA": "#2F6BFF", "FIA": "#12B8A6", "IUL": "#F3B61F"}

LEFT, RIGHT, TOP, BOTTOM = 0.07, 0.93, 0.94, 0.05
WIDTH = RIGHT - LEFT
BODY_SIZE = 7.4
LINE_H = 0.0138  # figure fraction per body text line
ROW_H = 0.0168  # figure fraction per table row
WRAP = 138  # characters per full-width body line

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": BODY_SIZE,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": RULE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.parse_math": False,
    }
)


def money(x: float, signed: bool = False, decimals: int = 0) -> str:
    if abs(x) < 0.5 * 10**-decimals:
        x = 0.0
    return f"{x:+,.{decimals}f}" if signed else f"{x:,.{decimals}f}"


class Page:
    """A letter page with a cursor that flows downwards in figure fractions."""

    def __init__(self, pdf: PdfPages, results: sm.SimpleModelResults, number: int):
        self.pdf = pdf
        self.fig: Figure = plt.figure(figsize=(8.5, 11.0))
        self.y = TOP
        m = results.market
        self.fig.text(
            LEFT,
            0.975,
            "Simple ALM hedging model: Pacific Life RILA, FIA, and IUL crediting segments",
            fontsize=9.5,
            weight="bold",
        )
        self.fig.text(
            LEFT,
            0.962,
            f"Illustrative model. Valuation {m.valuation_date.isoformat()}, maturity {m.maturity_date.isoformat()}, T = 1. "
            f"Insurer sign convention: positive P&L is a gain to the insurer.",
            fontsize=6.6,
            color=SLATE,
        )
        self.fig.text(
            RIGHT, 0.975, f"Page {number}", fontsize=7, ha="right", color=SLATE
        )

    def section(self, title: str, size: float = 9.2) -> None:
        self.y -= 0.004 * size / 9.2
        self.fig.text(LEFT, self.y, title, fontsize=size, weight="bold", va="top")
        self.y -= 0.016 * size / 9.2
        self.fig.add_artist(
            Line2D([LEFT, RIGHT], [self.y + 0.003, self.y + 0.003], color=RULE, lw=0.7)
        )
        self.y -= 0.005

    def paragraph(self, text: str, width: int = WRAP, size: float = BODY_SIZE) -> None:
        wrapped = textwrap.fill(text, width)
        self.fig.text(LEFT, self.y, wrapped, fontsize=size, va="top", linespacing=1.38)
        self.y -= LINE_H * (wrapped.count("\n") + 1) * size / BODY_SIZE + 0.006

    def bullets(
        self, items: Sequence[str], width: int = WRAP - 4, size: float = BODY_SIZE
    ) -> None:
        for item in items:
            wrapped = textwrap.fill(
                item, width, initial_indent="- ", subsequent_indent="  "
            )
            self.fig.text(
                LEFT, self.y, wrapped, fontsize=size, va="top", linespacing=1.38
            )
            self.y -= LINE_H * (wrapped.count("\n") + 1) * size / BODY_SIZE + 0.002
        self.y -= 0.004

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        col_widths: Sequence[float],
        left: float = LEFT,
        width: float = WIDTH,
        bold_last: bool = False,
        font: float = 6.6,
        advance: bool = True,
        row_scale: float = 1.0,
    ) -> float:
        height = ROW_H * (len(rows) + 1) * font / 6.6 * row_scale
        ax = self.fig.add_axes((left, self.y - height, width, height))
        ax.axis("off")
        table = ax.table(
            cellText=[list(r) for r in rows],
            colLabels=list(headers),
            colWidths=list(col_widths),
            cellLoc="right",
            bbox=(0, 0, 1, 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D6DEE8")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor(PALE)
                cell.set_text_props(weight="bold")
            if col == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.02
            if bold_last and row == len(rows):
                cell.set_text_props(weight="bold")
        if advance:
            self.y -= height + 0.012 * row_scale
        return height

    def axes(self, height: float, left: float = LEFT, width: float = WIDTH) -> Axes:
        """Axes box occupying ``height`` of the page, with room for a title above and labels below."""
        ax = self.fig.add_axes(
            (left + 0.045, self.y - height + 0.038, width - 0.045, height - 0.06)
        )
        return ax

    def close(self) -> None:
        self.fig.text(
            LEFT,
            0.024,
            textwrap.fill(
                "Generated by simple/build_report.py from simple_model.run_simple_model(); the notebook simple/simple_model.ipynb "
                "shows the same numbers. Illustrative model: not investment advice and not a Pacific Life position, price, or hedge.",
                165,
            ),
            fontsize=6.0,
            color=SLATE,
            va="top",
        )
        self.pdf.savefig(self.fig)  # type: ignore[no-untyped-call]
        plt.close(self.fig)


def section_1_terms(page: Page, r: sm.SimpleModelResults) -> None:
    m = r.market
    page.section("1. Product terms")
    page.paragraph(
        f"Three public product designs, each reduced to one {r.products[0].crediting} crediting segment on the "
        f"{sm.INDEX_NAME}, struck on {m.valuation_date.isoformat()} and maturing on {m.maturity_date.isoformat()}. "
        f"Market as of the valuation date: 1y Treasury {m.quoted_yield:.2%} quoted (continuous r = ln(1 + y) = {m.rate:.4%}), "
        f"dividend yield {m.dividend:.3%}, uniform 1y at-the-money-forward implied volatility {m.vol:.2%}, forward/spot {m.forward_factor():.4f}."
    )
    rows = []
    for p in r.products:
        protection = (
            f"{p.buffer:.0%} buffer" if p.buffer is not None else f"{p.floor:.0%} floor"
        )
        fee = f"{p.annual_fee * 100:.2f} pts / yr" if p.annual_fee else "none"
        rows.append(
            [
                p.product_name,
                f"${p.account_value:,.0f}",
                p.crediting,
                f"{p.participation:.0%}",
                f"{p.cap:.2%}",
                protection,
                fee,
            ]
        )
    page.table(
        [
            "Product",
            "Account value",
            "Crediting",
            "Participation",
            "Cap",
            "Protection",
            "Fee",
        ],
        rows,
        [0.38, 0.10, 0.15, 0.10, 0.07, 0.09, 0.11],
        font=6.2,
    )


def section_2_cash_flows(page: Page, r: sm.SimpleModelResults) -> None:
    page.section("2. Cash flows and embedded options")
    page.paragraph(
        "At inception the insurer receives the account value AV_0. At maturity it owes AV_T = AV_0 x (1 + credited rate), "
        "where credited rate = participation x option package payoff per $1 - annual deduction. The liability is the PV of AV_T and "
        "splits into a bond component (PV of returning AV_0), an option component (PV of the embedded option package), and a fee "
        "component (PV of the deduction): liability = bond + option - fee. Insurer value is the negative of the liability. "
        "The chart shows the credited rate; the first table shows the resulting account value cash flow at maturity for a few index "
        "outcomes; the second lists the option legs per product with their notional and PV."
    )
    page.y -= 0.004
    chart_h = 0.20
    ax = page.axes(chart_h, left=LEFT, width=0.42)
    idx = [100 * (x - 1) for x in r.payoff_grid]
    ax.plot(idx, idx, ls="--", color="#BBBBBB", lw=0.8, label="Index return")
    for name, rates in r.credited_rates.items():
        ax.plot(idx, [100 * c for c in rates], lw=1.6, color=COLORS[name], label=name)
    ax.axhline(0, color=RULE, lw=0.6)
    ax.set_xlabel("S&P 500 price return over the term (%)")
    ax.set_ylabel("Credited rate (%)")
    ax.set_title("Credited rate at maturity", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.grid(alpha=0.25)
    returns = (-0.20, -0.10, 0.0, 0.10, 0.20)
    rows = [
        [
            f"Index {ret:+.0%}",
            *[f"${p.account_value_at_maturity(ret):,.0f}" for p in r.products],
        ]
        for ret in returns
    ]
    rows.append(
        ["Credited rate at +20%", *[f"{p.credited_rate(0.20):.2%}" for p in r.products]]
    )
    page.table(
        ["AV paid at maturity", *[p.name for p in r.products]],
        rows,
        [0.40, 0.20, 0.20, 0.20],
        left=LEFT + 0.46,
        width=WIDTH - 0.46,
        advance=False,
    )
    page.y -= chart_h + 0.016
    leg_rows = []
    for v in r.valuations:
        for leg in v.legs:
            leg_rows.append(
                [
                    f"{v.product.name}: {leg.label}",
                    f"{leg.moneyness:.2%}",
                    f"{'+' if leg.quantity > 0 else '-'}${leg.notional:,.0f}",
                    f"${leg.greeks.price:,.2f}",
                ]
            )
        leg_rows.append(
            [f"{v.product.name}: option component", "", "", f"${v.option.price:,.2f}"]
        )
    page.table(
        [
            "Option leg (policyholder view)",
            "Strike (% of start index)",
            "Notional",
            "PV",
        ],
        leg_rows,
        [0.46, 0.20, 0.17, 0.17],
    )


def section_3_valuation(page: Page, r: sm.SimpleModelResults) -> None:
    m = r.market
    page.section("3. Valuation and Greeks")
    page.paragraph(
        "Liability PV per product and in total, with the interest budget (AV_0 - bond PV, what the premium earns at the risk-free rate) "
        "and the pricing margin (AV_0 - liability). Greeks are dollar sensitivities of the liability: per 1% index move, per 1%^2, "
        "per volatility point, and per basis point of the continuous rate. The insurer's exposure is the negative of each figure."
    )
    names = [v.product.name for v in r.valuations]
    totals = r.total
    rows = [
        [
            "Account value (premium received)",
            *[f"{v.product.account_value:,.2f}" for v in r.valuations],
            f"{r.total_account_value:,.2f}",
        ],
        [
            "Bond component PV",
            *[f"{v.bond.price:,.2f}" for v in r.valuations],
            f"{sum(v.bond.price for v in r.valuations):,.2f}",
        ],
        [
            "Option component PV",
            *[f"{v.option.price:,.2f}" for v in r.valuations],
            f"{r.option.price:,.2f}",
        ],
        [
            "Fee component PV",
            *[f"{v.fee.price:,.2f}" for v in r.valuations],
            f"{sum(v.fee.price for v in r.valuations):,.2f}",
        ],
        [
            "Liability PV = bond + option - fee",
            *[f"{v.liability_pv:,.2f}" for v in r.valuations],
            f"{totals.price:,.2f}",
        ],
        [
            "Interest budget = AV - bond",
            *[f"{v.interest_budget:,.2f}" for v in r.valuations],
            f"{sum(v.interest_budget for v in r.valuations):,.2f}",
        ],
        [
            "Pricing margin = AV - liability",
            *[f"{v.pricing_margin:,.2f}" for v in r.valuations],
            f"{sum(v.pricing_margin for v in r.valuations):,.2f}",
        ],
        [
            "Cap affordable with the full budget",
            *[f"{r.affordable_caps[n]:.2%}" for n in names],
            "",
        ],
    ]
    greek_labels = [
        "Delta ($ per 1% index)",
        "Gamma ($ per 1%^2)",
        "Vega ($ per vol pt)",
        "Rho ($ per bp)",
    ]
    per_product = [sm.dollar_greeks(v.total, m) for v in r.valuations]
    total_g = sm.dollar_greeks(totals, m)
    for label in greek_labels:
        rows.append(
            [
                label,
                *[money(g[label], decimals=2) for g in per_product],
                money(total_g[label], decimals=2),
            ]
        )
    option_g = sm.dollar_greeks(r.option, m)
    rows.append(
        [
            "Rho of the option component only ($ per bp)",
            *[
                money(sm.dollar_greeks(v.option, m)["Rho ($ per bp)"], decimals=2)
                for v in r.valuations
            ],
            money(option_g["Rho ($ per bp)"], decimals=2),
        ]
    )
    page.table(
        ["Liability view ($)", *names, "Portfolio"],
        rows,
        [0.36, 0.16, 0.16, 0.16, 0.16],
    )


def section_4_exposures(page: Page, r: sm.SimpleModelResults) -> None:
    m = r.market
    page.section("4. Net exposures")
    page.paragraph(
        "Portfolio exposures from the insurer's side, split by component, and the net option positions that the three products embed "
        "once offsetting legs are combined. The insurer is short the liability: it loses when the index rises, is long gamma and long vega "
        "(the short cap calls and the RILA buffer put give back more convexity than the long at-the-money calls provide), and its rate "
        "exposure sits almost entirely in the bond component."
    )
    comp = {
        "Option": sm.dollar_greeks(r.option, m),
        "Bond": sm.dollar_greeks(
            sum((v.bond for v in r.valuations), sm.ZERO_GREEKS), m
        ),
        "Fee": sm.dollar_greeks(sum((v.fee for v in r.valuations), sm.ZERO_GREEKS), m),
        "Total": sm.dollar_greeks(r.total, m),
    }
    labels = [
        "PV ($)",
        "Delta ($ per 1% index)",
        "Gamma ($ per 1%^2)",
        "Vega ($ per vol pt)",
        "Rho ($ per bp)",
    ]
    rows = [
        [
            label,
            money(-comp["Option"][label], decimals=2),
            money(-comp["Bond"][label], decimals=2),
            money(comp["Fee"][label], decimals=2),
            money(-comp["Total"][label], decimals=2),
        ]
        for label in labels
    ]
    page.table(
        ["Insurer view ($)", "-Option", "-Bond", "+Fee", "Insurer total"],
        rows,
        [0.36, 0.16, 0.16, 0.16, 0.16],
    )
    rep_rows = [
        [
            f"{'Long' if leg.net_notional > 0 else 'Short'} {leg.kind}s, strike {leg.moneyness:.2%} of start",
            f"{money(leg.net_notional, signed=True)}",
            f"{money(leg.cost, signed=True)}",
            leg.products,
        ]
        for leg in r.replication
    ]
    rep_rows.append(
        [
            "Net embedded option package",
            "",
            f"{money(r.static_replication_cost, signed=True)}",
            "all",
        ]
    )
    page.table(
        [
            "Net embedded option position (insurer is short this)",
            "Net notional ($)",
            "Policyholder PV ($)",
            "From",
        ],
        rep_rows,
        [0.46, 0.18, 0.18, 0.18],
        bold_last=True,
    )
    page.paragraph(
        f"Unhedged, the option component alone swings by ${r.option_unhedged_range:,.0f} across the stress scenarios in section 5; "
        f"the bond component adds about ${abs(next(x for x in r.stress if x.scenario == 'Rates +50 bp').bond_pnl):,.0f} per 50 bp of rates."
    )


def section_5_hedge(page: Page, r: sm.SimpleModelResults) -> None:
    m = r.market
    h = r.hedge
    page.section("5. Hedge recommendations")
    trades = [
        [
            "S&P 500 index futures (recommended initial hedge)",
            "Buy",
            f"${h.futures_notional:,.0f} exposure",
            "No premium",
            f"Offsets ${h.liability_delta * m.spot / 100:,.0f}/1% liability delta",
        ]
    ]
    for leg in r.replication:
        trades.append(
            [
                f"Static alternative: {leg.kind}s at {leg.moneyness:.2%}",
                "Buy" if leg.net_notional > 0 else "Sell",
                f"${abs(leg.net_notional):,.0f} notional",
                f"{'Pay' if leg.cost > 0 else 'Receive'} ${abs(leg.cost):,.0f}",
                f"Replicates {leg.products}",
            ]
        )
    trades.append(
        [
            "Static alternative: net",
            "",
            "",
            f"Pay ${r.static_replication_cost:,.0f} (= option PV)",
            "No rebalancing needed",
        ]
    )
    page.table(
        ["Instrument", "Action", "Size", "Cost", "Purpose"],
        trades,
        [0.31, 0.07, 0.17, 0.20, 0.25],
        font=6.0,
    )
    page.paragraph(
        f"Futures units: {h.futures_units:.2f} at a normalized index level of {m.spot:.0f}; convert to contracts with the live index level and multiplier. "
        "Futures margin, roll, and financing and the dealer spread on the static package are excluded.",
        size=6.6,
    )
    page.paragraph(
        f"Residual risks after the futures (insurer view): gamma {money(h.residual_gamma, True)} per 1%^2, vega {money(h.residual_vega, True)} "
        f"per vol point, rho {money(h.residual_rho, True)} per bp in total ({money(h.option_rho, True)} per bp on the options alone). "
        "Full-revaluation stress P&L, unhedged and with the futures:"
    )
    rows = [
        [
            s.scenario,
            money(s.bond_pnl, True),
            money(s.option_pnl, True),
            money(s.unhedged_pnl, True),
            money(s.hedge_pnl, True),
            money(s.option_hedged_pnl, True),
            money(s.hedged_pnl, True),
        ]
        for s in r.stress
    ]
    rows.append(
        [
            "Range (max - min)",
            "",
            money(r.option_unhedged_range),
            money(r.unhedged_range),
            "",
            money(r.option_hedged_range),
            money(r.hedged_range),
        ]
    )
    page.table(
        [
            "Scenario ($, insurer view)",
            "Bond",
            "Option",
            "Unhedged total",
            "Futures",
            "Option + futures",
            "Hedged total",
        ],
        rows,
        [0.30, 0.11, 0.11, 0.13, 0.11, 0.13, 0.11],
        bold_last=True,
    )
    reb_rows = [
        [
            x.scenario,
            f"{x.liability_dollar_delta:,.0f}",
            f"{x.futures_dollar_delta:,.0f}",
            f"{x.residual_dollar_delta:+,.0f}",
            f"{x.residual_notional:+,.0f}",
            "yes" if abs(x.residual_notional) > r.rebalance_threshold else "no",
        ]
        for x in r.rebalancing
    ]
    page.table(
        [
            "Rebalancing check: after",
            "Liability $/1%",
            "Futures $/1%",
            "Residual $/1%",
            "Residual exposure $",
            "Re-hedge?",
        ],
        reb_rows,
        [0.34, 0.13, 0.13, 0.13, 0.15, 0.12],
    )
    page.paragraph(
        f"Triggers: re-hedge when the residual index exposure exceeds ${r.rebalance_threshold:,.0f} "
        f"({r.rebalance_threshold / r.total_account_value:.0%} of account value), after a volatility move of 5 points or more, at least monthly, "
        f"and through the final month before {m.maturity_date.isoformat()} when the caps and buffer pin delta. Rate risk on the bond component "
        "should be managed with general-account asset duration, not with the equity hedge."
    )


def section_6_explanation(page: Page, r: sm.SimpleModelResults) -> None:
    page.section("6. Explanation of the results")
    for paragraph in sm.explanation(r):
        page.paragraph(paragraph)


def data_sources(page: Page, r: sm.SimpleModelResults) -> None:
    page.section("Data sources")
    page.bullets(
        [
            f"{s.name} [{s.kind}]: {s.value}. Source: {s.source}{'. ' + s.url if s.url else ''}. {s.note}"
            for s in r.sources
        ]
    )
    page.paragraph(
        "Observed inputs are read from the published series on the valuation date. Fitted or model inputs are choices made from observed data: "
        "the continuous-rate conversion ignores day count, the single volatility ignores skew and term structure, and the index level is a normalization.",
        size=6.8,
    )


def considerations(page: Page, r: sm.SimpleModelResults) -> None:
    page.section("Things we considered")
    page.bullets(r.considered)
    page.section("Things we have not considered")
    page.bullets(r.exclusions)
    page.section("Things we can consider in the next steps")
    page.bullets(r.next_steps)


def build_report(results: sm.SimpleModelResults, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        page = Page(pdf, results, 1)
        section_1_terms(page, results)
        section_2_cash_flows(page, results)
        page.close()

        page = Page(pdf, results, 2)
        section_3_valuation(page, results)
        section_4_exposures(page, results)
        page.close()

        page = Page(pdf, results, 3)
        section_5_hedge(page, results)
        page.close()

        page = Page(pdf, results, 4)
        section_6_explanation(page, results)
        data_sources(page, results)
        page.close()

        page = Page(pdf, results, 5)
        considerations(page, results)
        page.close()
    return output


def build_one_page(results: sm.SimpleModelResults, output: Path) -> Path:
    """Condensed single page with the same six sections and a scope footer."""
    r, m, h = results, results.market, results.hedge
    small, tiny = 6.4, 5.9
    wrap = 158
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        page = Page(pdf, r, 1)
        page.fig.texts[2].set_text("One page")
        page.y = 0.950

        page.section("1. Product terms", size=7.8)
        rows = [
            [
                p.product_name,
                f"${p.account_value:,.0f}",
                f"{p.crediting}, {p.participation:.0%} part.",
                f"{p.cap:.2%}",
                f"{p.buffer:.0%} buffer"
                if p.buffer is not None
                else f"{p.floor:.0%} floor",
                f"{p.annual_fee * 100:.2f} pts / yr" if p.annual_fee else "none",
            ]
            for p in r.products
        ]
        page.table(
            [
                "Product (S&P 500 price return)",
                "Account value",
                "Crediting",
                "Cap",
                "Protection",
                "Fee",
            ],
            rows,
            [0.36, 0.11, 0.23, 0.08, 0.10, 0.12],
            font=5.8,
            row_scale=0.8,
        )
        page.y += 0.004
        page.paragraph(
            f"Market {m.valuation_date.isoformat()}: 1y Treasury {m.quoted_yield:.2%} quoted (continuous {m.rate:.4%}), dividend yield {m.dividend:.3%}, "
            f"uniform 1y ATMF vol {m.vol:.2%}. Maturity {m.maturity_date.isoformat()}, T = 1.",
            width=wrap,
            size=tiny,
        )

        page.section("2. Cash flows and embedded options", size=7.8)
        page.paragraph(
            "Insurer receives AV_0 now and owes AV_T = AV_0 x (1 + credited rate) at maturity, where credited rate = option package payoff per $1 "
            "- annual deduction. Liability = bond (PV of AV_0) + option (PV of the package) - fee (PV of the deduction); insurer value is its negative.",
            width=wrap,
            size=small,
        )
        chart_h = 0.112
        ax = page.axes(chart_h, left=LEFT, width=0.40)
        idx = [100 * (x - 1) for x in r.payoff_grid]
        ax.plot(idx, idx, ls="--", color="#BBBBBB", lw=0.7)
        for name, rates in r.credited_rates.items():
            ax.plot(
                idx, [100 * c for c in rates], lw=1.4, color=COLORS[name], label=name
            )
        ax.axhline(0, color=RULE, lw=0.5)
        ax.set_xlabel("Index return over the term (%)", fontsize=5.8)
        ax.set_ylabel("Credited rate (%)", fontsize=5.8)
        ax.tick_params(labelsize=5.5)
        ax.legend(frameon=False, fontsize=5.6, loc="upper left")
        ax.grid(alpha=0.25)
        rep_rows = [
            [
                f"{'Long' if leg.net_notional > 0 else 'Short'} {leg.kind}s at {leg.moneyness:.2%}",
                money(leg.net_notional, signed=True),
                money(leg.cost, signed=True),
                leg.products,
            ]
            for leg in r.replication
        ]
        rep_rows.append(
            [
                "Net embedded option package",
                "",
                money(r.static_replication_cost, signed=True),
                "all",
            ]
        )
        page.table(
            [
                "Embedded option legs",
                "Net notional $",
                "PV $",
                "From",
            ],
            rep_rows,
            [0.44, 0.20, 0.16, 0.20],
            left=LEFT + 0.43,
            width=WIDTH - 0.43,
            font=5.8,
            row_scale=0.8,
            bold_last=True,
            advance=False,
        )
        page.y -= chart_h + 0.004

        page.section("3. Valuation and Greeks (liability view, $)", size=7.8)
        names = [v.product.name for v in r.valuations]
        per = [sm.dollar_greeks(v.total, m) for v in r.valuations]
        tot = sm.dollar_greeks(r.total, m)
        rows = [
            [
                "Bond PV",
                *[money(v.bond.price) for v in r.valuations],
                money(sum(v.bond.price for v in r.valuations)),
            ],
            [
                "Option PV",
                *[money(v.option.price) for v in r.valuations],
                money(r.option.price),
            ],
            [
                "Fee PV",
                *[money(v.fee.price) for v in r.valuations],
                money(sum(v.fee.price for v in r.valuations)),
            ],
            [
                "Liability PV = bond + option - fee",
                *[money(v.liability_pv) for v in r.valuations],
                money(r.total.price),
            ],
            [
                "Pricing margin = AV - liability",
                *[money(v.pricing_margin) for v in r.valuations],
                money(sum(v.pricing_margin for v in r.valuations)),
            ],
        ]
        for label in (
            "Delta ($ per 1% index)",
            "Gamma ($ per 1%^2)",
            "Vega ($ per vol pt)",
            "Rho ($ per bp)",
        ):
            rows.append(
                [
                    label,
                    *[money(g[label], decimals=2) for g in per],
                    money(tot[label], decimals=2),
                ]
            )
        page.table(
            ["", *names, "Portfolio"],
            rows,
            [0.36, 0.16, 0.16, 0.16, 0.16],
            font=5.8,
            row_scale=0.8,
        )

        page.section("4. Net exposures (insurer view)", size=7.8)
        page.paragraph(
            f"The insurer is short the liability: it loses ${h.liability_delta * m.spot / 100:,.0f} per 1% index rise, is long gamma "
            f"({money(h.residual_gamma, True)} per 1%^2) and long vega ({money(h.residual_vega, True)} per vol point) because the short cap calls and the RILA "
            f"buffer put give back more convexity than the long ATM calls provide, and carries {money(h.residual_rho, True)} per bp of rate exposure, "
            f"of which only {money(h.option_rho, True)} comes from the options; the rest is the ${sum(v.bond.price for v in r.valuations):,.0f} bond component. "
            f"Unhedged, the option component swings ${r.option_unhedged_range:,.0f} across the stresses below.",
            width=wrap,
            size=small,
        )

        page.section("5. Hedge recommendations", size=7.8)
        page.paragraph(
            f"Buy S&P 500 index futures with ${h.futures_notional:,.0f} of index exposure ({h.futures_units:.2f} units at level {m.spot:.0f}); no premium, margin and roll excluded. "
            f"Static alternative: buy the net option package above for ${r.static_replication_cost:,.0f} (= option PV, dealer spread excluded), no rebalancing. "
            f"Re-hedge the futures when residual index exposure exceeds ${r.rebalance_threshold:,.0f} ({r.rebalance_threshold / r.total_account_value:.0%} of AV; "
            f"a 5% index move or a 5-point vol move breaches it), at least monthly, and through the final month to maturity. Manage bond-component rate risk with asset duration.",
            width=wrap,
            size=small,
        )
        rows = [
            [
                s_.scenario,
                money(s_.bond_pnl, True),
                money(s_.option_pnl, True),
                money(s_.unhedged_pnl, True),
                money(s_.hedge_pnl, True),
                money(s_.option_hedged_pnl, True),
                money(s_.hedged_pnl, True),
            ]
            for s_ in r.stress
            if s_.scenario not in ("Vol -5 pts", "Rates -50 bp")
        ]
        rows.append(
            [
                "Range (max - min)",
                "",
                money(r.option_unhedged_range),
                money(r.unhedged_range),
                "",
                money(r.option_hedged_range),
                money(r.hedged_range),
            ]
        )
        page.table(
            [
                "Full-revaluation stress ($, insurer view)",
                "Bond",
                "Option",
                "Unhedged total",
                "Futures",
                "Option + futures",
                "Hedged total",
            ],
            rows,
            [0.30, 0.11, 0.11, 0.13, 0.11, 0.13, 0.11],
            font=5.8,
            row_scale=0.8,
            bold_last=True,
        )

        page.section("6. Explanation of the results", size=7.8)
        by = {v.product.name: v for v in r.valuations}
        rila_put = next(leg for leg in by["RILA"].legs if leg.kind == "put")
        down20 = next(x for x in r.stress if x.scenario == "Index -20%")
        up20 = next(x for x in r.stress if x.scenario == "Index +20%")
        vol_up = next(x for x in r.stress if x.scenario == "Vol +5 pts")
        page.bullets(
            [
                f"The RILA's {by['RILA'].product.cap:.2%} cap costs ${by['RILA'].option.price:,.0f}, less than the FIA's ${by['FIA'].option.price:,.0f} for a "
                f"{by['FIA'].product.cap:.0%} cap, because the policyholder has written a 90% put worth ${-rila_put.greeks.price:,.0f} that funds a further-out cap call.",
                f"The ${sum(v.interest_budget for v in r.valuations):,.0f} interest budget on ${r.total_account_value:,.0f} pays for ${r.option.price:,.0f} of options, "
                f"leaving ${sum(v.pricing_margin for v in r.valuations):,.0f} of margin before expenses and dealer spreads; caps are therefore set by rates and volatility.",
                f"The futures remove the first-order move (index -20%: {money(down20.option_pnl, True)} unhedged vs {money(down20.option_hedged_pnl, True)} hedged; "
                f"+20%: {money(up20.option_pnl, True)} vs {money(up20.option_hedged_pnl, True)}) and leave the insurer long gamma and vega, so large moves and higher "
                f"vol help ({money(vol_up.option_hedged_pnl, True)} for +5 vol points) while a vol fall costs. Rate P&L is mostly the bond component and is an "
                "asset-duration question. A single fitted vol likely flatters the RILA margin relative to the observed skew.",
            ],
            width=wrap - 2,
            size=tiny,
        )

        page.section("Data sources, scope, and next steps", size=7.8)
        page.paragraph(
            "Sources: 1y Treasury yield, observed, FRED DGS1 (fred.stlouisfed.org/series/DGS1); dividend yield, observed, Shiller/Yale "
            "(econ.yale.edu/~shiller/data.htm); 1y ATMF vol, fitted, Cboe Option EOD Summary (datashop.cboe.com/option-eod-summary); "
            "VIX cross-check, observed, Cboe VIX historical data; index level normalized to 100.",
            width=wrap,
            size=tiny,
        )
        page.paragraph(
            "Considered: one dated indexed term per product; option-leg decomposition; bond/option/fee split with insurer sign; analytic Greeks; observed vs fitted inputs; "
            "static replication and futures delta hedge; full-revaluation stresses; rebalancing triggers.",
            width=wrap,
            size=tiny,
        )
        page.paragraph(
            "Not considered: other IUL charges, monthly mechanics, insurance coverage, premiums, withdrawals/surrender, RILA early exit/MVA, optional benefits, loans, adviser fees, "
            "future years, real valuation and hedging costs.",
            width=wrap,
            size=tiny,
        )
        page.paragraph(
            "Next steps: SPX skew and term structure; full Treasury and dividend curves; excluded IUL and RILA mechanics; policyholder behavior and renewals; "
            "cost-adjusted static vs dynamic comparison; gamma/vega hedges via a surface model; back-test through the October 2026 selloff.",
            width=wrap,
            size=tiny,
        )
        page.close()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / "simple-model-report.pdf")
    parser.add_argument(
        "--one-page-output",
        type=Path,
        default=here / "simple-model-one-page.pdf",
    )
    args = parser.parse_args(argv)
    results = sm.run_simple_model()
    for path in (
        build_report(results, args.output),
        build_one_page(results, args.one_page_output),
    ):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
