from ultralytics import YOLO
import cv2
from PIL import Image


def train():
    model = YOLO("yolov8n.pt")

    model.train(data="YOLO/data.yaml", epochs=60, imgsz=640, save=True, patience=5)


def load_model():
    source = "YOLO/train/images/5e596c17-chatpate49_jpg.rf.i8L1iH7sfn9ZN94LFKyq.jpg"
    model = YOLO("runs/detect/train-2/weights/best.pt")
    results = model.predict(
        source=source,
        conf=0.01,
        save=True,
        show=False,
    )

    for result in results:
        xywh = result.boxes.xywh
        xyxy = result.boxes.xyxy

        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]

        confs = result.boxes.conf

        print("Classes:", names)
        print("Confidence:", confs)


load_model()
# if __name__ == "__main__":
#     train()
