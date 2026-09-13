from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "visdrone_yolo"
    / "data.yaml"
)

MODEL_NAME = "yolo26n.pt"

RUN_NAME = "baseline_dev"


def main():

    print("=" * 60)
    print("AeroVision-Edge Baseline Training")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)
    print("Dataset:", DATA_YAML)
    print("Model:", MODEL_NAME)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),

        # Development baseline
        epochs=3,

        # Standard YOLO input size
        imgsz=640,

        # CPU-friendly batch size
        batch=2,

        # Explicitly use CPU
        device="cpu",

        # Conservative Windows setting
        workers=0,

        # Keep experiment outputs organized
        project=str(PROJECT_ROOT / "runs"),
        name=RUN_NAME,

        # Don't accidentally overwrite an earlier experiment
        exist_ok=False,

        # Reproducibility
        seed=42,
    )

    print("\nBaseline training completed.")


if __name__ == "__main__":
    main()