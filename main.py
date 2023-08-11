from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=50, imgsz=640)