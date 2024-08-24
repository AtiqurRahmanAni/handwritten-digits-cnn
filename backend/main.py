from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Image(BaseModel):
    image: List[List[float]]


transform = transforms.Compose([
    transforms.Resize(size=(28, 28), antialias=True),
])


class CNN(nn.Module):
    def __init__(self, l1=1000, l2=500) -> None:
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.droupout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=l1)
        self.fc2 = nn.Linear(in_features=l1, out_features=l2)
        self.fc3 = nn.Linear(in_features=l2, out_features=10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1_2(self.relu(self.conv1_1(x)))))
        x = self.pool(self.relu(self.conv2_2(self.relu(self.conv2_1(x)))))
        x = self.pool(self.relu(self.conv3_2(self.relu(self.conv3_1(x)))))
        x = self.droupout(self.relu(self.fc1(x.view(x.shape[0], -1))))
        x = self.droupout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


activations = {}


def get_activation(layer_name):
    def hook(module, input, output):
        activations[layer_name] = output.detach().squeeze(0)

    return hook


cnn = CNN(l1=1024, l2=256)
cnn.load_state_dict(torch.load("./train_model.pt",
                    map_location="cpu", weights_only=True))
cnn.eval()
softmax = nn.Softmax(dim=1)

conv_layers = ['conv1_1', 'conv1_2',
               'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2']
cnn.conv1_1.register_forward_hook(get_activation('conv1_1'))
cnn.conv1_2.register_forward_hook(get_activation('conv1_2'))

cnn.conv2_1.register_forward_hook(get_activation('conv2_1'))
cnn.conv2_2.register_forward_hook(get_activation('conv2_2'))

cnn.conv3_1.register_forward_hook(get_activation('conv3_1'))
cnn.conv3_2.register_forward_hook(get_activation('conv3_2'))


def convert_to_255(tensor):
    min_vals = tensor.min()
    max_vals = tensor.max()
    scaled = ((tensor - min_vals) / (max_vals - min_vals)) * 255
    scaled = scaled.round().clamp(0, 255)
    return scaled


@app.get("/")
async def root():
    return {"message": "API is working"}


@app.post("/api/classify")
async def classify(image: Image):
    img = transform(torch.tensor(image.image).unsqueeze(0))

    torchvision.utils.save_image(img, "./test.png")

    with torch.no_grad():
        probas = softmax(cnn(img.unsqueeze(0)))
        predicted = torch.argmax(probas).item()
    for conv_layer in conv_layers:
        for i in range(activations[conv_layer].shape[0]):
            activations[conv_layer][i] = convert_to_255(
                activations[conv_layer][i])

        activations[conv_layer] = activations[conv_layer].to(torch.uint8)

    return {"probas": probas.view(-1).tolist(), "predicted": predicted, "activations": {
        "conv1_1": activations['conv1_1'].tolist(),
        "conv1_2": activations['conv1_2'].tolist(),
        "conv2_1": activations['conv2_1'].tolist(),
        "conv2_2": activations['conv2_2'].tolist(),
        "conv3_1": activations['conv3_1'].tolist(),
        "conv3_2": activations['conv3_2'].tolist(),
    }}
