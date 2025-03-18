import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import MobileViTForImageClassification


model_path = "/content/mobilevit_model.pth"


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileViTForImageClassification.from_pretrained(
        "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model(model_path)


def classify_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).logits
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# image_path = "/content/apple.png"

# classify_image(image_path, model, device)
