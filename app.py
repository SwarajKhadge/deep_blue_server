from flask import Flask, request, jsonify
import torch
import io
from PIL import Image

app = Flask(__name__)

# Load the YOLOv5 model (this will download the model if needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# Optionally adjust confidence threshold if desired, e.g.:
# model.conf = 0.25

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Run detection
    results = model(img)
    detections = []
    # Convert results to a pandas DataFrame
    df = results.pandas().xyxy[0]
    for index, row in df.iterrows():
        # Convert 'person' to 'Human'
        label = row['name']
        if label.lower() == 'person':
            label = 'Human'
        detections.append({
            'label': label,
            'confidence': float(row['confidence']),
            'bbox': [float(row['xmin']), float(row['ymin']),
                     float(row['xmax']), float(row['ymax'])]
        })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    # Run on port 5000 (adjust host/IP as needed)
    app.run(host='0.0.0.0', port=5000)
