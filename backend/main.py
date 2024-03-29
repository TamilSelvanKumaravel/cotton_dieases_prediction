from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = ["https://cotton-dieases-prediction-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("C:/Users/tamilselvan/cotton/backend/model/cotton_best.h5")
CLASS_NAMES = ["diseased_cotton_leaf", "diseased_cotton_plant", "fresh_cotton_leaf", "fresh_cotton_plant"]


@app.get("/ping")
async def ping():
  return "Hello,I am alive"


def read_file_as_image(data) -> np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  image = read_file_as_image(await file.read())
  img_batch = np.expand_dims(image, 0)
  resized_input_data = tf.image.resize(img_batch, (28, 28))#changed line

  prediction = MODEL.predict(resized_input_data)
  predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
  confidence = np.max(prediction[0])
  return {
      'class': predicted_class,
      'confidence': float(confidence)
  }


if __name__ == "__main__":
  uvicorn.run(app, host='localhost', port=8000)
