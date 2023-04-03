import json
import uuid
import cv2
import io
import uvicorn

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
import sys
sys.path.append("..")

from ml import predicting_cat

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to fastAPI"}

@app.post("/prediction")
async def prediction(file: UploadFile = File(...)):
    img = Image.open(file.file)
    labels, probs = predicting_cat.transfer_learning_find_breed(img=img)
    values = {
        "first_prediction": f"{labels[0]}",
        "second_prediction": f"{labels[1]}",
        "third_prediction": f"{labels[2]}"
    }
    return values

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)

#to run uvicorn main:app --reload