
import torch
import onnx
H, W = 416, 640
EXPORT_PATH = 'yolov5n.onnx'
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n').cuda()

# Export to ONNX
inputs = torch.ones((1, 3, H, W)).cuda()
torch.onnx.export(model.model.model, inputs, EXPORT_PATH, input_names=['images'], export_params=True, do_constant_folding=True, opset_version=11)

# Add metadata
onnx_model = onnx.load(EXPORT_PATH)
meta = onnx_model.metadata_props.add()
meta.key = 'names'
classes = {i: CLASSES[i] for i in range(len(CLASSES))}
meta.value = str(classes)
onnx.save(onnx_model, EXPORT_PATH)
print('Done yay!')
