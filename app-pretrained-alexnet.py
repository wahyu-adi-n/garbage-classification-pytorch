import torch
import torch.nn as nn
from torchvision import transforms
import streamlit as st
from PIL import Image
import io
import time


class GarbageClassifier():

    MODEL_PATH = 'model_scripted.pt'
    LABELS_PATH = 'labels.txt'
    NUM_CLASSES = 12

    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH, num_classes=NUM_CLASSES):
        self.model = model_path
        self.labels_path = labels_path
        self.num_classes = num_classes

    def loadImage(self):
        image = st.sidebar.file_uploader(
            label='Pick Your Image as an Input',
            type=['jpg', 'png', 'png'])
        if image is not None:
            img = image.getvalue()
            return img
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
        return proba, category_id


if __name__ == '__main__':
    st.write("""
        # Web Apps - Garbage Classification
    """)
    st.sidebar.header('Image Input')
    picture = st.sidebar.camera_input("Take a Picture")
    garbage = GarbageClassifier()
    image = garbage.loadImage() if picture is None else picture.getvalue()
    model = garbage.loadModel()
    categories = garbage.loadCategories()
    if image is None:
        st.write('No image yet, please upload!')
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image)
            result = st.button('Click This Button to Classify')
        with col2:
            num_result = st.sidebar.slider(
                'Num Clasess', 1, 12, 1)
            if result:
                st.write('### Result: ')
                image = Image.open(io.BytesIO(image))
                t = time.time()
                proba, category_id = garbage.predict(model, categories, image)
                time = time.time() - t
                for i in range(num_result):
                    st.write("""
                        #### Class: {} 
                        #### Confidence: {:.2f} %
                    """.format(categories[category_id[i]], proba[i].item()*100))
                st.write("Inference Time: {:.3f}s".format(time))
