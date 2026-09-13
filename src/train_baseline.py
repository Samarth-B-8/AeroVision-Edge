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


def main():

    print("Project root:", PROJECT_ROOT)
    print("Dataset YAML:", DATA_YAML)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    # Load pretrained lightweight YOLO model
    model = YOLO("yolo26n.pt")

    # Tiny smoke-test training run
    results = model.train(
        data=str(DATA_YAML),
        epochs=1,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        project="runs/aerovision",
        name="baseline_smoke_test",
        exist_ok=True
    )

    print("\nTraining completed.")
    print(results)


if __name__ == "__main__":
    main()