from ml import predicting_cat
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import Query
from pydantic import BaseModel
import sys
import uvicorn
sys.path.append("..")

app = FastAPI()


class Message(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"message": "Welcome to fastAPI"}


@app.post("/prediction", responses={404: {"model": Message}})
async def prediction(file: UploadFile = File(...), algorithm: str = Query(...)):
    # try to open image
    try:
        img = Image.open(file.file)
    # exception for image not opening
    except IOError:
        return JSONResponse(status_code=404, content={"message": "Error opening image"})
    # general exception
    except Exception as e:
        return JSONResponse(status_code=404, content={"message": "Something went wrong"})
    # if the algorithm given is good we send the top 3 predictions
    if (algorithm == "Transfer Learning(resnet50)" or algorithm == "My Model"):
        labels, probs = predicting_cat.transfer_learning_find_breed(
            img=img, model_str=algorithm)
        values = {
            "first_prediction": f"{labels[0]}",
            "second_prediction": f"{labels[1]}",
            "third_prediction": f"{labels[2]}"
        }
        return values
    # if the algorithm is wrong we give error
    else:
        return JSONResponse(status_code=404, content={"message": "Model given is wrong"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000)

# to run uvicorn main:app --reload
