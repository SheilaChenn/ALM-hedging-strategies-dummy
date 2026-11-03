# Simple ALM hedging model

Illustrative model of one S&P 500 point-to-point crediting term (valuation
2026-10-26, maturity 2027-10-26) for three public product designs with
simplified terms: Pacific Protective Growth RILA ($100,000; 16.25% cap, 10%
buffer), Pacific Index Advisory FIA ($100,000; 10% cap, 0% floor), and the
Pacific Horizon IUL 2 1-Year High Cap Indexed Account ($10,000 indexed account;
12% cap, 0% floor before a simplified 0.80-point annual deduction). Closed-form
Black-Scholes on a flat continuous rate (`ln(1.0447)` from the FRED DGS1 yield),
a flat dividend yield (Shiller data), and one fitted 1y ATMF volatility (Cboe
option data). Observed inputs are distinguished from fitted/model assumptions,
and the exclusions are listed explicitly in the module, notebook, and reports.
Not a Pacific Life position, price, or hedge, and not investment advice.

- `simple/simple_model.py`: Black-Scholes legs with analytic Greeks; each
  product as option legs plus a fixed deduction; bond + option - fee liability
  decomposition with the insurer sign convention; dollar Greeks; static option
  replication; index-futures delta hedge; full-revaluation stresses; rebalancing
  triggers; `run_simple_model()` returns everything.
- `simple/simple_model.ipynb`: eight sections (inputs and sources, terms and
  payoffs, valuation, Greeks, replication and delta hedge, stresses,
  recommendation, exclusions) with saved outputs.
- `simple/build_report.py` renders two PDFs from the same `run_simple_model()`
  output (1 product terms, 2 cash flows and embedded options, 3 valuation and
  Greeks, 4 net exposures, 5 hedge recommendations, 6 explanation of the
  results, then data sources, what was considered, what was not, and next
  steps): `simple/simple-model-report.pdf` (five pages) and
  `simple/simple-model-one-page.pdf` (one-page summary).
- `tests/test_simple_model.py` pins the terms, amounts, decomposition, fee,
  bond + option reconciliation, finite-difference Greeks, hedge, stresses, and
  checks that the reports build.

```bash
python -m pytest tests/test_simple_model.py
python simple/build_report.py
jupyter nbconvert --to notebook --execute --inplace simple/simple_model.ipynb
```

Dependencies: numpy, pandas, matplotlib, pytest, jupyter.
