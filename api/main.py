from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2 as cv
import uvicorn

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/epoch50.keras")

CLASS_NAMES = ["Acne", "Dry", "Normal", "Oily"]

# Converts the uploaded image to a numpy array


def preprocess_image(file):
    img_bytes = BytesIO(file)
    img = Image.open(img_bytes)
    img = np.array(img)

    resized = cv.resize(img, (224, 224))
    return resized


@app.get("/")
async def root():
    return {"Webpage is alive"}


@app.post("/level")
async def skin_level(file: UploadFile = File(...)):

    image = preprocess_image(await file.read())
    print('File read')

    prediction = MODEL.predict(np.expand_dims(image, axis=0))
    confidence = round(100 * (np.max(prediction[0])), 2)

    class_index = np.argmax(prediction)
    skin_type = CLASS_NAMES[class_index]

    return {
        'Skin Type': skin_type,
        'Confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
