import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
import os
import torch.optim as optim
from urllib import request
import torchvision.transforms
from gmpy2 import random_state
from matplotlib import pyplot as plt
import random
import cv2
import glob
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.pyplot import subplots
from torchvision import transforms, datasets
import datetime

# # 이미지 출력 테스트
# car_path_test = glob.glob('../img/Car/*.jpeg')
# car_list = car_path_test[:100]
#
# fig, ax = plt.subplots(5, 5, figsize=(15, 15))
#
# for i in range(5):
#     for j in range(5):
#         idx = i * 5 + j
#         img = Image.open(car_list[idx])
#         ax[i][j].imshow(img)
#         ax[i][j].axis('off')
#
# plt.tight_layout()
# plt.show()
# print(car_list)

# 이미지 가져오기
train_path = glob.glob('../img/Car/*.jpeg')
target_path = glob.glob('C:/Users/oxo97/Downloads/background/1/BG-20k/train/*.jpg')
train_image = [cv2.imread(p) for p in train_path[:100]]
target_image = [cv2.imread(q) for q in target_path[:100]]

# print(train_image[0].shape)
#
# test_torch = torch.tensor(train_image[0])
#
# print(test_torch.shape)


transform = transforms.Compose([
    transforms.ToPILImage(),  # OpenCV로 읽은 이미지 반환값 numpy -> PIL 이미지 객체 변환
    transforms.CenterCrop(224),  # 이미지 중앙 기준 224, 224 자르기
    transforms.ToTensor()  # PIL 이미지를 Tensor 로 변환, 픽셀 값을 [0,1] 사이로 정규화
])

# 이미지 변환
train_images = [transform(img) for img in train_image]
target_images = [transform(img) for img in target_image]

print(len(train_images))
print(len(target_images))

# 라벨 지정
train_labels = torch.zeros(len(train_images), dtype=torch.long)
target_labels = torch.ones(len(target_images), dtype=torch.long)

# 훈련 / 타겟 하나로 묶기
all_images = torch.stack(train_images + target_images)
all_labels = torch.cat([train_labels, target_labels])
train_dataset = TensorDataset(all_images, all_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()        # 활성화 함수
        self.pool = nn.MaxPool2d(2)  # 이미지 풀링

        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 224, 224 -> 112, 112

        self.conv2 = nn.Conv2d(8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 112, 112 -> 56, 56

        self.conv3 = nn.Conv2d(12, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 56, 56 -> 28, 28

        self.conv4 = nn.Conv2d(16, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 28, 28 -> 14, 14

        self.conv5 = nn.Conv2d(12, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # 14, 14 -> 7, 7

        self.fc1 = nn.Linear(8 * 7 * 7, 2)  # 최종 출력

    def forward(self, x):   # 순차적 으로 신행
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = self.pool(self.relu(self.conv4(out)))
        out = self.pool(self.relu(self.conv5(out)))

        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        return out

# GPU가 사용 가능 하면 GPU를 사용, 아니면 CPU를 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

# 하이퍼 파라미터 설정 / 손실함수, 훈련횟수, 학습율. 최적화 모듈 선택
loss_fn = nn.CrossEntropyLoss()
n_epochs = 100
learning_rate = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 루프를 통해 설정한 하이퍼 파라미터를 적용하여 훈련 수행
def training_loop(n_epochs, model, optimizer, loss_fn):
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'{datetime.datetime.now()} | Epoch {epoch} | Train Loss: {running_loss / len(train_loader):.4f}')

# 호출
training_loop(n_epochs, model, optimizer, loss_fn)

model.eval()

# 위 선언한 transform 과 겹치지 않게 재정의
inference_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 이미지 분류 테스트
def visualize_prediction(img_path, model, transform):
    img = Image.open(img_path).convert("RGB")
    input_tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # [1, 2]
        probs = torch.softmax(output, dim=1)  # [1, 2]
        pred = torch.argmax(probs, dim=1).item()  # 0 or 1

    label = "Car" if pred == 0 else "Background"
    plt.imshow(img)
    plt.title(f"Predicted: {label}")
    plt.axis("off")
    plt.show()

# 이미지 분류 테스트 하기 위한 태스트 세트 이미지 가져오기
test_img_paths = glob.glob('../img/Car/*.jpeg')

for path in test_img_paths[1000:1400]:
    visualize_prediction(path, model, transform)