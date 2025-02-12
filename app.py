from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from transformers import MobileViTForImageClassification

app = Flask(__name__)
CORS(app)

# Load the model
model_path = "mobilevit_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViTForImageClassification.from_pretrained(
    "apple/mobilevit-small", num_labels=2, ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def classify_image(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor).logits
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Image Classification API'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            image = Image.open(request.files['file'])
        elif 'url' in request.json:
            image_url = request.json['url']
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
            else:
                return jsonify({'error': 'Could not fetch image from URL'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400

        predicted_class = classify_image(image)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/model', methods=['POST'])
def model_request():
    try:
        data = request.get_json()  # Get the JSON payload
        print("Received JSON Payload:", data)  # Print payload to the console
        
        return jsonify({
            'message': 'Received JSON payload successfully',
            'received_data': data
        })
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)