from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Loading Model
model = tf.keras.models.load_model("main_model.h5")  

app = FastAPI()

# Image processing
def preprocess_image(image_bytes, target_size=(256, 256)):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    return np.array(image) / 255.0

@app.post("/predict")
async def predict(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1 = preprocess_image(await image1.read())
        img2 = preprocess_image(await image2.read())

        input_pair = np.concatenate([img1, img2], axis=-1)
        input_pair = np.expand_dims(input_pair, axis=0)  # (1, 256, 256, 6)

        prediction = model.predict(input_pair)
        mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8).tolist()

        return JSONResponse(content={"status": "success", "mask": mask})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
