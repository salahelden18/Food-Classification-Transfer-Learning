from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import tensorflow as tf
import json

# Read the JSON file containing the class names
with open('./class_names.json', 'r') as f:
    class_names = json.load(f)

app = FastAPI()

MODEL = tf.saved_model.load("../food_model_3")

# class_names = [
#
# ]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = ImageOps.fit(image, (224, 224))  # Resize and crop the image to (224, 224)
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    image_batch = tf.cast(image_batch, tf.float32)

    # call the model on the test data
    result = MODEL.signatures['serving_default'](tf.constant(image_batch))

    prediction = result['output_layer']
    prediction_index = tf.argmax(prediction[0]).numpy()
    max_value = tf.reduce_max(prediction, axis=1)
    max_value = max_value.numpy()[0]
    max_value = max_value * 100
    print(max_value)
    print(prediction_index)

    return {
        'result': class_names[prediction_index],
        'percent': f'{max_value:.2f}%'
    }

    # return image

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)