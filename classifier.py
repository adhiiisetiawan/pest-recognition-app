import torch
import string
from torch import nn
from PIL import Image
from torchvision import models, transforms


class InsectPestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = models.mobilenet_v2()
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 102)
        )
        
    def forward(self, x):
        return self.mobilenet(x)


def predict(image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = InsectPestClassifier().to(device)
    weights = torch.load('model/cutmix_model_ft_epoch_241-260.pth', map_location=torch.device('cpu'))
    model.load_state_dict(weights['model_state_dict'])
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch_t)

    with open('classes.txt') as f:
        classes = [string.capwords(line[3:]).strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
