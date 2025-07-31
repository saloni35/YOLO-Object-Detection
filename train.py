import logging, os, sys
from pathlib import Path
from ultralytics import YOLO


log_file = "training_error.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to train a YOLOv8 model using custom YOLO training dataset.
    """
    logger.info("Starting YOLOv8 training process...")
    try:
        # Load YOLOv8 pretrained model to start with
        model = YOLO("yolov8s.pt")

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

        model.eval()
        logger.info(f"Model architecture:\n{str(model)}")

        # Save the trained model
        os.makedirs("saved_model", exist_ok=True)
        model.save("saved_model/best.pt")
        print("Training complete. Model saved as 'saved_model/best.pt'.")

    except Exception as e:

        logger.exception("Training failed due to an unexpected error.")
        print("\n An error occurred during training. Please check the log file for details:")
        print(f"Log file: {Path(log_file).resolve()}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()