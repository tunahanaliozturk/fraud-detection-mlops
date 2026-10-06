"""Card-transaction fraud detection: synthetic data, point-in-time features, training, serving and monitoring."""

import os

# MLflow sends anonymous usage events unless told not to. Nothing in this package should talk to the internet.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "true")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
