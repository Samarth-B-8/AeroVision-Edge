from pathlib import Path
import cv2


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "visdrone"

OUTPUT_DATA = PROJECT_ROOT / "data" / "processed" / "visdrone_yolo"


# --------------------------------------------------
# VisDrone → YOLO class mapping
# --------------------------------------------------

CLASS_MAP = {
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
    11: 10  # others
}

def convert_box(x, y, width, height, image_width, image_height):
    """
    Convert VisDrone pixel coordinates into
    normalized YOLO coordinates.
    """

    x_center = x + width / 2
    y_center = y + height / 2

    x_center /= image_width
    y_center /= image_height

    width /= image_width
    height /= image_height

    return x_center, y_center, width, height

def convert_annotation(annotation_path, image_path, output_label_path):
    """
    Convert one VisDrone annotation file into YOLO format.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]

    yolo_lines = []

    with open(annotation_path, "r") as file:
        for line in file:
            values = line.strip().split(",")

            if len(values) != 8:
                continue

            x = int(values[0])
            y = int(values[1])
            width = int(values[2])
            height = int(values[3])

            score = int(values[4])
            category = int(values[5])

            # Ignore VisDrone ignored regions
            if category == 0:
                continue

            # Ignore annotations not considered valid for evaluation
            if score == 0:
                continue

            # Ignore degenerate boxes
            if width <= 0 or height <= 0:
                continue

            if category not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[category]

            x_center, y_center, box_width, box_height = convert_box(
                x,
                y,
                width,
                height,
                image_width,
                image_height
            )

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_label_path, "w") as file:
        file.write("\n".join(yolo_lines))

def convert_split(split_name):
    """
    Convert one VisDrone split.
    """

    source_dir = RAW_DATA / f"VisDrone2019-DET-{split_name}"

    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"

    output_image_dir = OUTPUT_DATA / split_name / "images"
    output_label_dir = OUTPUT_DATA / split_name / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.jpg"))

    print(f"\nConverting {split_name} split...")
    print(f"Images found: {len(image_files)}")

    for image_path in image_files:

        annotation_path = annotation_dir / f"{image_path.stem}.txt"

        if not annotation_path.exists():
            print(f"Missing annotation: {image_path.name}")
            continue

        # Copy image
        destination_image = output_image_dir / image_path.name

        if not destination_image.exists():
            destination_image.write_bytes(image_path.read_bytes())

        # Convert annotation
        destination_label = output_label_dir / f"{image_path.stem}.txt"

        convert_annotation(
            annotation_path,
            image_path,
            destination_label
        )

    print(f"Finished {split_name} split.")

if __name__ == "__main__":
    convert_split("train")
    convert_split("val")

    print("\nDataset conversion complete.")