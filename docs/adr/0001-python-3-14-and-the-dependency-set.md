# 1. Python 3.14 and the dependency set

Status: Accepted, 2026-10-06

## Context

The house rule is the newest stable Python, unless a key package has no wheel for it. The packages that usually lag
are the compiled ones: LightGBM, SHAP and its numba/llvmlite stack, and anything MLflow pulls in. Every dependency
must also be under a permissive licence at every depth, checked by `tools/license_audit.py`.

## Decision

We use Python 3.14. Checked on PyPI on 2026-10-06: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1,
shap 0.52.0, numba 0.68.0, llvmlite 0.50.0 and pyarrow 25.0.1 all ship cp314 wheels; LightGBM 4.7.0 ships a
platform wheel that is not tied to a Python version. We take `mlflow-skinny` 3.16.1 rather than full `mlflow`, plus
SQLAlchemy and Alembic for the SQLite tracking store.

SQLAlchemy is held at the 2.0 line (2.0.54, still maintained) instead of 2.1.3. MLflow 3.16's ORM mappings use the
`noload` loader strategy, which SQLAlchemy 2.1 deprecates with a warning that it "produces incorrect results". We
would rather run MLflow on the SQLAlchemy line it was built against than silence that warning.

## Consequences

- 77 distributions in the locked tree, all permissive (MIT, BSD, Apache-2.0, PSF-2.0, and MPL-2.0 for certifi, tqdm
  and pathspec). llvmlite is "Apache-2.0 WITH LLVM-exception"; the audit treats an SPDX exception as the extra
  permission it is.
- Compiled wheels vendor OpenMP runtimes (scikit-learn on Linux) under the GCC Runtime Library Exception, and the
  image installs `libgomp1` for LightGBM. Neither is a Python distribution, so the audit does not see them; both
  allow use in proprietary software.
- Two warnings come from MLflow's own code and are filtered in pytest, each with a comment: `codecs.open` (deprecated
  in 3.14) and a type-hint notice MLflow raises about its own chat-agent schema at import. Everything else runs with
  warnings as errors.
- Starlette's `TestClient` now asks for `httpx2`. The tests call the app through `httpx.ASGITransport` instead
  (`tests/conftest.py`), which needs no new dependency.
- The image is 1.36 GB. The virtual environment inside it is 860 MB, led by llvmlite (173 MB), pyarrow (156 MB)
  and scipy (110 MB), measured with `du` in the built image.

## Alternatives

- **Python 3.13.** The fallback if any compiled package had lacked a 3.14 wheel. None did.
- **Full `mlflow`.** Adds the tracking server, Flask, matplotlib and more. We need the client, the registry and a
  SQLite store, which the skinny package plus SQLAlchemy covers. It would win if we wanted the MLflow UI shipped too.
- **SQLAlchemy 2.1 with the warning filtered.** Newer, but the warning is about wrong results, not style.
