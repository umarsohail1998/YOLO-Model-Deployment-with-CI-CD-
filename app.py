from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import numpy as np
import os

# Load model once at startup
MODEL_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
model = YOLO(MODEL_WEIGHTS)

app = FastAPI(title="YOLOv8 FastAPI Inference")

@app.get("/")
def index():
    return {"status": "ok", "message": "YOLOv8 inference API. POST /predict with form-file 'file'."}

@app.post("/predict")
async def predict(file: UploadFile = File(...), return_image: bool = False):
    """
    Upload an image (form-data: file). Optional query param return_image=true to get annotated PNG.
    Returns JSON detections by default.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    # Run inference
    results = model(img)  # results is a Results object; it holds list-like items (one per image)
    r = results[0]  # single image

    # JSON output
    detections = []
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy.tolist()[0]  # [x1,y1,x2,y2]
            conf = float(box.conf.tolist()[0])
            cls = int(box.cls.tolist()[0])
            name = model.names[cls] if hasattr(model, "names") else str(cls)
            detections.append({
                "class_id": cls,
                "class_name": name,
                "confidence": conf,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
            })

    response = {
        "detections": detections,
        "model": {"weights": MODEL_WEIGHTS},
        "img_size": {"width": img.width, "height": img.height}
    }

    if return_image:
        # Draw annotated image bytes and return PNG
        annotated = r.plot()  # returns numpy array (BGR)
        # r.plot returns np.ndarray in RGB or BGR depending on version. Convert safely:
        if isinstance(annotated, np.ndarray):
            # If colors are BGR, convert to RGB by reversing channels if needed
            if annotated.shape[2] == 3:
                annotated = annotated[:, :, ::-1]  # BGR->RGB safe op (if already RGB it'll reverse but still viewable)
            pil_img = Image.fromarray(annotated)
        else:
            pil_img = Image.fromarray(np.array(annotated))

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
