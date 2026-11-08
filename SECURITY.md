# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub's
[Private vulnerability reporting](https://github.com/heinricitorgau/Chebyshev-Wavelet-Core/security/advisories/new):
go to the **Security** tab of this repository and choose **Report a vulnerability**.

Please include:

- the affected file(s) and version or commit,
- steps to reproduce or a proof of concept,
- the impact you expect.

You can expect an acknowledgement within 7 days. Once a fix is ready, we will publish an
advisory and credit you unless you prefer to stay anonymous.

## Scope

In scope:

- the MATLAB code in this repository,
- the Python paper-trading path (`chebyshev_wavelet_py/`, `run_sandbox.py`), especially anything
  that could bypass the `paper_only` / `submit_orders` safeguards or leak credentials.

Out of scope:

- vulnerabilities in MATLAB, Python, `ib_async`, Interactive Brokers, or the upstream data APIs
  (please report those to their maintainers),
- trading losses: this project is research software and is not intended for live trading.

## Supported versions

Only the latest commit on `main` receives fixes.
