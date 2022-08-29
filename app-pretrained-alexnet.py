import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st
from PIL import Image
import io
import os


class GarbageClassifier():

    MODEL_PATH = 'model_scripted.pt'
    LABELS_PATH = 'labels.txt'
    NUM_CLASSES = 12

    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH, num_classes=NUM_CLASSES):
        self.model = model_path
        self.labels_path = labels_path
        self.num_classes = num_classes

    def loadImage(self):
        upload = st.sidebar.file_uploader(
            label='Pick Your Image as an Input', type=['jpg', 'png', 'png'])
        if upload is not None:
            img = upload.getvalue()
            st.image(img)
            return Image.open(io.BytesIO(img))
        return None

    def loadModel(self):
        model = torch.jit.load(self.model, map_location=torch.device('cpu'))
        model.eval()
        return model

    def loadCategories(self):
        with open(self.labels_path, "r") as f:
            categories = [s.strip() for s in f.readlines()]
            return categories

    def predict(self, model, categories, image):
        image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = image_transforms(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = nn.functional.softmax(output[0], dim=0)

        proba, category_id = torch.topk(probabilities, len(categories))
        for i in range(proba.size(0)):
            st.write(categories[category_id[i]], proba[i].item())


if __name__ == '__main__':
    st.write("""
        # Web Apps - Garbage Classification
    """)
    st.sidebar.header('Image Input')
    garbage = GarbageClassifier()
    image = garbage.loadImage()
    model = garbage.loadModel()
    categories = garbage.loadCategories()
    result = st.button('Click This Button to Classify')

    if image is not None:
        if result:
            st.write('Calculating results...')
            garbage.predict(model, categories, image)
    else:
        st.write('No image yet, please upload!')
