from pathlib import Path
import sys

import mlflow
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_prediction.config import PROJECT_ROOT  # noqa: E402

print(f"Environment is set up correctly at {PROJECT_ROOT}")
