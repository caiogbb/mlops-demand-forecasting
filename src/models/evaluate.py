from __future__ import annotations

import json

from src.config import METRICS_PATH


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Training metrics were not generated.")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    best_model = metrics["best_model"]

    print(f"Best model: {best_model['name']}")

    print(
        json.dumps(
            best_model["metrics"],
            indent=2,
        )
    )

    print(f"Promoted to champion: {metrics['promoted_to_champion']}")


if __name__ == "__main__":
    main()
