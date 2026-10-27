# Simple ALM hedging model

Illustrative, synthetic model of the embedded options in one year of RILA, FIA, and IUL crediting. Closed-form Black-Scholes only. Not a Pacific Life production model and not investment advice.

- `simple/simple_model.py`: Black-Scholes legs with analytic Greeks; RILA, FIA, and IUL as lists of vanilla options; cap solver; full-revaluation stress table; futures delta hedge.
- `simple/simple_model.ipynb`: six short sections with saved outputs.
- `simple/build_report.py` renders `simple/simple-model-report.pdf` from the same `run_simple_model()` output.
- `tests/test_simple_model.py`: put-call parity, RILA payoff edge cases, finite-difference Greeks, cap solver, and hedge.

```bash
python -m pytest tests/test_simple_model.py
python simple/build_report.py
jupyter nbconvert --to notebook --execute --inplace simple/simple_model.ipynb
```

Dependencies: numpy, scipy, matplotlib, pytest, jupyter.
