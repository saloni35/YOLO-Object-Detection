import io, os, logging
from typing import Dict, List
from collections import Counter

from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image
from ultralytics import YOLO
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)

conf_thresh = 0.5

# Global variable to store the loaded model
model_flower: YOLO = None
model_coco_yolo8: YOLO = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for managing the lifespan of the FastAPI application.
    Loads the YOLO model on startup and performs cleanup on shutdown.
    """
    global model_flower
    global model_coco_yolo8
    model_path = os.path.join(os.path.dirname(__file__), "./custom_model/best.pt")
    try:

        #Configure Logging to File
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True) # Create 'logs' directory if it doesn't exist
        log_file_path = os.path.join(log_dir, "app.log")

        if logger.hasHandlers():
            logger.handlers.clear()

        # Set the logging level for this specific logger (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
        logger.setLevel(logging.INFO) # Or logging.DEBUG for more verbose logs

        # Create a FileHandler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO) # Set level for this handler

        formatter = logging.Formatter('%(levelname)s - %(message)s')

        # Set the formatter for the file handler
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        model_flower = YOLO(model_path)
        model_coco_yolo8 = YOLO("yolov8s.pt")
        logger.info(f"YOLO model loaded successfully from: {model_path} and default model loaded.")
        logger.info(f"Custom model device: {next(model_flower.parameters()).device}")
        logger.info(f"Default model device: {next(model_coco_yolo8.parameters()).device}")

    except Exception as e:
        logger.error(f"ERROR: Could not load YOLO model from {model_path} or default model. Reason: {e}")
        model_flower = None
        model_coco_yolo8 = None

    yield # Application startup is complete, and the application can now receive requests.

    model_flower = None
    model_coco_yolo8 = None
    logger.info("FastAPI application shutting down.")



app = FastAPI(lifespan=lifespan)


async def detect_and_count_objects(img_contents: bytes) -> Dict[str, str] | Dict[str, List[Dict[str, str | int]]]:
    """ Detect objects in the provided image bytes and count occurrences of each object class."""
    if model_coco_yolo8 is None or model_flower is None:
        raise HTTPException(status_code=500, detail="Object detection models are not loaded. Server might be misconfigured.")

    img_rgb = Image.open(io.BytesIO(img_contents))

    # Perform inference on the image
    # 'results_flower' will be a list of Results objects (one per image if you pass a list of images)
    results_flower = model_flower(img_rgb, conf=conf_thresh)
    results_yolo8 = model_coco_yolo8(img_rgb, conf=conf_thresh)

    # Process results_flower for the first (and only) image
    if not results_flower and not results_yolo8:
        logger.info("No detections found.")
        return {"objects": "No detections found."}

    count_flowers = Counter()
    count_coco_objects = Counter()
    if results_flower:
        result = results_flower[0] # Get the Results object for the current image

        # Extract detected class IDs and their names
        detected_class_ids = result.boxes.cls.tolist() # Convert tensor to list
        class_names = model_flower.names # Get the mapping from class ID to class name (e.g., {0: 'person', 1: 'bicycle', ...})

        # Count objects by class label
        for class_id in detected_class_ids:
            class_name = class_names[int(class_id)]
            count_flowers[class_name] += 1

    if results_yolo8:
        result = results_yolo8[0]

        # Extract detected class IDs and their names
        detected_class_ids = result.boxes.cls.tolist() # Convert tensor to list
        class_names = model_coco_yolo8.names # Get the mapping from class ID to class name (e.g., {0: 'person', 1: 'bicycle', ...})

        # Count objects by class label
        for class_id in detected_class_ids:
            class_name = class_names[int(class_id)]
            count_coco_objects[class_name] += 1

    data_flower_obj = [{"object_name": name, "object_count": count} for name, count in count_flowers.items()]
    data_coco_obj = [{"object_name": name, "object_count": count} for name, count in count_coco_objects.items()]
    total_data = data_flower_obj + data_coco_obj

    return {"objects": total_data}


async def verify_image(file: File = File(...)):
    try:
        # Validate file type provided by the client
        # For more robust validation might involve reading magic bytes.
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

        # Read the file content into memory
        contents = await file.read()

        # Basic image validation using Pillow
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify if it's a valid image
            # If verify() passes, the image is likely valid.

            logger.info(f"Received image: {file.filename}, Format: {image.format}, Size: {image.size}")
        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            # If the image is not valid, raise an HTTPException
            raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

        # 4. Return a success response
        return contents

    except HTTPException as e:
        # Re-raise HTTPException to be handled by FastAPI's default exception handler
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Upload a file and check the object name and object count in return
    """
    try:
        img_content = await verify_image(file=file)

        return await detect_and_count_objects(img_content)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")





