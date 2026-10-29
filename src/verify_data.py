
import zipfile
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def read_dataset(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        splits = {}
        eval_hashes = {}

        for split in ["train", "val", "test"]:
            images = sorted(
                n for n in names
                if n.startswith(f"images/{split}/")
                and n.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            )

            boxes = Counter()
            backgrounds = 0

            for image_name in images:
                stem = Path(image_name).stem
                label_name = f"labels/{split}/{stem}.txt"

                if label_name not in names:
                    raise ValueError(
                        f"Missing label: {label_name}"
                    )

                lines = z.read(label_name).decode().splitlines()

                if not lines:
                    backgrounds += 1

                for line in lines:
                    p = line.split()
                    if len(p) != 5:
                        raise ValueError(f"Invalid label: {label_name}")

                    cls = int(p[0])
                    coords = [float(v) for v in p[1:]]

                    if cls not in range(4):
                        raise ValueError(f"Invalid class: {cls}")

                    if not all(0 <= v <= 1 for v in coords):
                        raise ValueError("Invalid coordinates")

                    if coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError("Invalid box dimensions")

                    boxes[cls] += 1

                if split in ["val", "test"]:
                    eval_hashes[image_name] = hashlib.sha256(
                        z.read(image_name)
                    ).hexdigest()
                    eval_hashes[label_name] = hashlib.sha256(
                        z.read(label_name)
                    ).hexdigest()

            splits[split] = {
                "images": len(images),
                "backgrounds": backgrounds,
                "boxes_per_class": dict(boxes)
            }

    return splits, eval_hashes

v1, v1_eval = read_dataset(ROOT / "data/rdd2022_v1.zip")
v2, v2_eval = read_dataset(ROOT / "data/rdd2022_v2.zip")

if v1_eval != v2_eval:
    raise ValueError("Validation/test differs between versions")

if [v1[s]["images"] for s in ["train","val","test"]] != [300,75,75]:
    raise ValueError("Unexpected v1 sizes")

if [v2[s]["images"] for s in ["train","val","test"]] != [1000,75,75]:
    raise ValueError("Unexpected v2 sizes")

report = {
    "v1": v1,
    "v2": v2,
    "evaluation_unchanged": True
}

output = ROOT / "reports/data_validation.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, indent=2))

print("Dataset validation passed!")
print(json.dumps(report, indent=2))
