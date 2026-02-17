from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# Load model once at startup
model = YOLO("best.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    detections = []
    original_img = results[0].plot()  # image with bounding boxes

    # Convert full image with boxes to base64
    full_buffer = io.BytesIO()
    Image.fromarray(original_img).save(full_buffer, format="PNG")
    full_base64 = base64.b64encode(full_buffer.getvalue()).decode()

    for box in results[0].boxes:
        cls = int(box.cls)
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop detected region
        cropped = image.crop((x1, y1, x2, y2))

        crop_buffer = io.BytesIO()
        cropped.save(crop_buffer, format="PNG")
        crop_base64 = base64.b64encode(crop_buffer.getvalue()).decode()

        detections.append({
            "field_name": label,
            "coordinates": [x1, y1, x2, y2],
            "cropped_image_base64": crop_base64
        })

    return {
        "full_image_with_boxes_base64": full_base64,
        "detections": detections
    }
