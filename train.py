from ultralytics import YOLO
import os

# Load YOLOv8 model (you can also use 'yolov8n.pt', 'yolov8m.pt', etc.)
model = YOLO("yolov8s.pt")  # or 'yolov8n.pt', 'yolov8m.pt', etc.


data_path = "dataset/data.yaml"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Training Data files not found at: {data_path}")

# Train using your Roboflow dataset
results = model.train(
    data=data_path,  # path to the data.yaml
    epochs=50,
    imgsz=640,
    batch=16,
)

# Save the trained model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/best.pt")
print("Training complete. Model saved as 'saved_model/best.pt'.")
