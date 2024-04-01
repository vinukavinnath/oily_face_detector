# Import libraries to access google cloud bucket object
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "oily_face_model_0"
class_names = ["Acne", "Dry", "Normal", "High"]
model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    # Used to download the model from cloud
    storage_client = storage.Client()
    # Gets the bucket
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


def predict(request):
    # Must be called once
    # To make sure model is empty add a condition
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model_VGG_epochs_50.h5",
            "/tmp/model_VGG_epochs_50.h5"
        )
        # Loads the model
        model = tf.keras.load_model("/tmp/model_VGG_epochs_50.h5")

        image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))  # image resizing
    )

    image = image/255  # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}
