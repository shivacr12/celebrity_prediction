import os
from pathlib import Path
import sys
from PIL import Image
import pickle
import torch
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File,HTTPException
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms


app = FastAPI()

image_dir_save = os.path.join(Path(sys.path[0]).resolve(), 'uploaded_images')
# Load the saved model and preprocessing steps
model_filename = os.path.join(Path(sys.path[0]).resolve(),'model_artifacts','celebrity_classifier.pkl')

with open(model_filename, 'rb') as f:
    saved_data = pickle.load(f)

clf = saved_data['classifier']
label_dict = saved_data['label_dict']

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


@app.post("/predict/")
async def predict_image(image: UploadFile = File(...)):
    try:
       
        contents = await image.read()
        image.filename = 'uploaded_image.jpg'  

        with open(f'{image_dir_save}/{image.filename}', 'wb') as f:
            f.write(contents)

        embedding = extract_face_embedding(image=f'{image_dir_save}/{image.filename}')

        prediction = clf.predict(embedding)[0]

        return JSONResponse(content={'prediction': label_dict[prediction]})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)