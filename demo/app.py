import torch
from torch import nn
import torchvision
import gradio as gr

# Define and load my resnet50 model
model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    # Add dropout layer with 50% probability
    nn.Dropout(0.5),
    # Add a linear layer in order to deal with 5 classes
    nn.Linear(num_ftrs, 5),
)

model.load_state_dict(
    torch.load("model/final_model_state_dict.pth", map_location=torch.device("cpu"))
)
model.eval()

# Define the labels
labels = ["bird", "cat", "dog", "horse", "sheep"]


# Define the predict function
def predict(inp):
    inp = torchvision.transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = model(inp)
        # Map prediction to label
        prediction = labels[prediction.argmax()]
    return prediction


# Define the gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=[["demo/input_imgs/cat.jpeg"], ["demo/input_imgs/dog.jpeg"]],
    description="This is a demo of a resnet50 model trained on COCO dataset, which can classify 5 classes: bird, cat, dog, horse, sheep."
)

demo.launch()
