from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
import httpx
from PIL import Image
import io
from ultralytics import YOLO
from collections import Counter
from contextlib import asynccontextmanager # Import asynccontextmanager
import os


conf_thresh = 0.5


# Global variable to store the loaded model
custom_model: YOLO = None
default_model: YOLO = None

# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for managing the lifespan of the FastAPI application.
    Loads the YOLO model on startup and performs cleanup on shutdown.
    """
    global custom_model
    global default_model
    model_path = os.path.join(os.path.dirname(__file__), "./custom_model/best.pt")
    try:
        custom_model = YOLO(model_path)
        default_model = YOLO("yolov8s.pt")
        print(f"YOLO model loaded successfully from: {model_path} and default model loaded.")
    except Exception as e:
        print(f"ERROR: Could not load YOLO model from {model_path} or default model. Reason: {e}")
        custom_model = None
        default_model = None

    yield # Application startup is complete, and the application can now receive requests.

    # --- Cleanup (optional, runs on shutdown) ---
    # For YOLO, explicit cleanup might not be strictly necessary,
    # but for other resources (e.g., database connections), this is where you'd close them.
    print("FastAPI application shutting down.")
    # You could add yolo_model.close() if the library supported it for explicit resource release.


app = FastAPI(lifespan=lifespan)

#app.mount("/custom_model", StaticFiles(directory="custom_model"), name="custom_model")


async def detect_and_count_objects(contents: bytes):

    # Convert the image from BGR (OpenCV default) to RGB (for Matplotlib display if needed)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global custom_model
    global default_model
    img_rgb = Image.open(io.BytesIO(contents))

    # Perform inference on the image
    # 'results' will be a list of Results objects (one per image if you pass a list of images)
    results = custom_model(img_rgb, conf=conf_thresh)
    results_default = default_model(img_rgb, conf=conf_thresh)

    # Process results for the first (and only) image
    if not results and not results_default:
        print("No detections found.")
        return {"objects": "No detections found."}

    object_counts = Counter()
    default_object_counts = Counter()
    if results:
        result = results[0] # Get the Results object for the current image

        # Extract detected class IDs and their names
        detected_class_ids = result.boxes.cls.tolist() # Convert tensor to list
        class_names = custom_model.names # Get the mapping from class ID to class name (e.g., {0: 'person', 1: 'bicycle', ...})

        # Count objects by class label

        for class_id in detected_class_ids:
            class_name = class_names[int(class_id)]
            object_counts[class_name] += 1

    if results_default:
        result_default = results_default[0]

        # Extract detected class IDs and their names
        detected_class_ids_default = result_default.boxes.cls.tolist() # Convert tensor to list
        class_names_default = default_model.names # Get the mapping from class ID to class name (e.g., {0: 'person', 1: 'bicycle', ...})
        # Count objects by class label
        for class_id in detected_class_ids_default:
            class_name = class_names_default[int(class_id)]
            default_object_counts[class_name] += 1

    data = [{"object_name": name, "object_count": count} for name, count in object_counts.items()]
    data_default = [{"object_name": name, "object_count": count} for name, count in default_object_counts.items()]
    total_data = data + data_default

    return {"objects": total_data}


async def verify_image(file: File = File(...)):
    try:
        # 1. Validate file type (optional but recommended)
        # You can check the media_type provided by the client.
        # Be aware that clients can send incorrect media types, so
        # more robust validation might involve reading magic bytes.
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

        # 2. Read the file content
        # For small to medium files, reading into memory is fine.
        # For very large files, consider streaming to disk or processing in chunks.
        contents = await file.read()

        # Optional: Basic image validation using Pillow
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify if it's a valid image
            # If verify() passes, the image is likely valid.
            # You can also perform operations like image.size, image.format here.
            print(f"Received image: {file.filename}, Format: {image.format}, Size: {image.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not process image: {e}")



        # 4. Return a success response
        return contents

    except HTTPException as e:
        # Re-raise HTTPException to be handled by FastAPI's default exception handler
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.get("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    """
    Upload a file and check the object name and object count in return
    """
    try:
        img_content = await verify_image(file=file)

        return await detect_and_count_objects(img_content)


    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Error from external service: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")





