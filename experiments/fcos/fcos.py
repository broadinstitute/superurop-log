import torchvision
import torch
import cv2

LR = 1e-4
EPOCHS = 20

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


# freeze entire model first
for param in model.parameters():
    param.requires_grad = False

# or just freeze the resnet50 backbone
# for param in model.backbone.parameters():
#     param.requires_grad = False

# unfreeze regression head, and final transform module
for param in model.head.regression_head.parameters():
    param.requires_grad = True
for param in model.transform.parameters():
    param.requires_grad = True

# training loop boilerplate
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# dataloader = # define dataloader

for i, epoch in range(EPOCHS):
    for images, targets in dataloader:
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()
        loss.backward()
        optimizer.step()