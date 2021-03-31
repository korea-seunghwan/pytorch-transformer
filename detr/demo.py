import torch as th
import requests
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

model = th.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
model.eval()
model = model.to(device)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# print(model)

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSfwl_iPUW1MlDriTXNQDoiutIuilmQOpITLg&usqp=CAU"

img = Image.open(requests.get(url, stream=True).raw).resize((800, 600))

img_tens = transform(img).unsqueeze(0).to(device)

with th.no_grad():
    output = model(img_tens)

# print(output['pred_boxes'][0].shape)
img2 = img.copy()
drw = ImageDraw.Draw(img2)
pred_logits=output['pred_logits'][0]
pred_boxes=output['pred_boxes'][0]

for logits, box in zip(pred_logits, pred_boxes):
    cls = logits.argmax()
    if cls >= len(CLASSES):
        continue
    label = CLASSES[cls]
    # print(label)
    box = box.cpu() * th.Tensor([800, 600, 800, 600])
    x, y, w, h = box
    x0, x1 = x-w//2, x+w//2
    y0, y1 = y-h//2, y+h//2
    drw.rectangle([x0, y0, x1, y1], outline='red', width=1)
    drw.text((x, y), label, fill='red')
    # break

# print(img2.format)
img2.save('result.png', format='png')