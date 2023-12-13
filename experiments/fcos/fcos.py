import torchvision
import torch
import cv2

# load vanilla(non-pretrained) FCOS model from torchvision
model = torchvision.models.detection.fcos_resnet50_fpn()
model.eval()

# run inference on sample image
img = cv2.imread("BBBC038_sample_img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = torch.from_numpy(img)
img = img.float()
img = img.permute(2, 0, 1)
img = img.unsqueeze(0)

out = model(img)
out = out[0]

bounding_boxes, scores = out["boxes"], out["scores"]

print(list(zip(bounding_boxes, scores)))