from pathlib import Path
from ultralytics import YOLO
import multiprocessing

DATA_YML = Path(__file__).resolve().parent.parent / "dataset" / "drone" / "data.yml"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(DATA_YML),
        epochs=150,
        patience=50,
        imgsz=640,
        batch=16,
        workers=4,
        cache="ram",
        seed=42,
        name="pothole_yolo26n_drone",
        device="mps"
    )

    with open(Path(results.save_dir) / "results.json", "w") as f:
        f.write(results.to_json())
