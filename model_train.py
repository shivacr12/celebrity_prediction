import os
from pathlib import Path
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Define paths
data_dir = os.path.join(Path(sys.path[0]).resolve(), 'data')
model_dir = os.path.join(Path(sys.path[0]).resolve(), 'model_artifacts')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'validation')

# Define face detection and recognition models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define transformations for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def extract_face_embedding(image_path):
    image = preprocess(Image.open(image_path))
    embedding = resnet(image.unsqueeze(0).to(device)).detach().cpu().numpy()
    return embedding

def extract_face_embeddings(images_dir):
    embeddings = []
    labels = []
    for label, celebrity_dir in enumerate(sorted(os.listdir(images_dir))):
        print('label, celebrity_dir : ',label, celebrity_dir)
        celebrity_path = os.path.join(images_dir, celebrity_dir)
        for image_file in os.listdir(celebrity_path):
            image_path = os.path.join(celebrity_path, image_file)
            embedding = extract_face_embedding(image_path = image_path)
            embeddings.append(embedding)
            labels.append(label)
    return np.array(embeddings).reshape(len(embeddings), -1), np.array(labels)


train_embeddings, train_labels = extract_face_embeddings(train_dir)
validation_embeddings, validation_labels = extract_face_embeddings(validation_dir)

print('model fitting on training data')
# Train a classifier
clf = make_pipeline(StandardScaler(), SVC())
clf.fit(train_embeddings, train_labels)

# Evaluate the classifier
print('model evaluation')
predictions = clf.predict(validation_embeddings)
accuracy = accuracy_score(validation_labels, predictions)
print('Accuracy:', accuracy)

if accuracy > 0.9:
    # Retrain on both train and validation sets
    print('model retrain')
    combined_embeddings = np.concatenate([train_embeddings, validation_embeddings])
    combined_labels = np.concatenate([train_labels, validation_labels])
    clf.fit(combined_embeddings, combined_labels)

    # Save the model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = os.path.join(model_dir, 'celebrity_classifier.pkl')
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)

    print('Model saved successfully as', model_filename)
else:
    print('Accuracy is not above 90%, model not saved.')