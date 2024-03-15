import io
import pickle
import torch
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File,HTTPException
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import uuid
import numpy as np
import os
from pathlib import Path
import sys
import cv2
from pydantic import BaseModel
from PIL import Image


app = FastAPI()

image_dir_save = os.path.join(Path(sys.path[0]).resolve(), 'uploaded_images')
# Load the saved model and preprocessing steps
with open('model_relics/celebrity_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def extract_face_embedding(image):

    image = preprocess(Image.open(image))

    embedding = resnet(image.unsqueeze(0).to(device)).detach().cpu().numpy()

    return embedding




# @app.post("/predict/")
# async def predict(image: UploadFile = File(...)):

#     # Read image file
#     contents = await image.read()
#     img = Image.open(io.BytesIO(contents)).convert("RGB")

#     # Extract face embeddings
#     embedding = extract_face_embedding(img)

#     # Make prediction
#     prediction = clf.predict(embedding.reshape(1, -1))

#     # Convert NumPy array to Python list for JSON serialization
#     prediction = prediction.tolist()

#     return JSONResponse(content={"predicted_celebrity_label": prediction[0]})


# class Analyzer(BaseModel):
#     filename: str
#     img_dimensions: str
#     encoded_img: str

@app.post("/upload/")
async def upload_file(image: UploadFile = File(...)):
    try:
       
        contents = await image.read()
        image.filename = 'uploaded_image.jpg'  
        with open(f'{image_dir_save}/{image.filename}', 'wb') as f:
            f.write(contents)

        embedding = extract_face_embedding(image=f'{image_dir_save}/uploaded_image.jpg')

        prediction = str(clf.predict(embedding)[0])

        return JSONResponse(content={'prediction': prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)