from __future__ import annotations

import os
import sys
from pathlib import Path

# Robust local API launcher for source-tree runs. This avoids shell/path issues
# where the `uvicorn` executable may be resolved outside the virtualenv.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONPATH", str(SRC))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "paisa.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(SRC),
    )
