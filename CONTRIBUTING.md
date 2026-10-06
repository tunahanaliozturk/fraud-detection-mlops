# Contributing

Thanks for looking. A few things make a change easy to accept.

## Setting up

```sh
uv sync
uv run pytest          # offline, about 20 s once the bytecode is compiled
```

## What a change needs

- `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy src tests` clean. CI runs the same, and
  pytest treats warnings as errors.
- A test that fails without the change. Know which line of the change makes it go red.
- **Anything that touches features** keeps them point-in-time: a feature reads the history before `add()` is called,
  never after. `tests/test_features.py` must stay green without loosening it.
- **Anything that can change a metric** (features, model parameters, the split, the cost model, the generator) comes
  with `uv run fraud evaluate` before and after in the pull request, and the files under `docs/results/` regenerated
  in the same change. Choose settings on the validation window, never on the test window (ADR 0004).
- Numbers in the README come from a command in this repository and name the hardware they ran on.
- An ADR for any decision someone could reasonably have made differently. ADRs are not edited after they are merged;
  a reversal gets a new number.
- New dependencies permissively licensed at every depth; `uv run python tools/license_audit.py` checks. Run
  `uv lock` after editing `pyproject.toml` and commit the lock file.

## Commits

Conventional commit subjects (`fix(features): ...`, `feat(serve): ...`), one logical change per commit.
