import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import requests
import json

model = models.mobilenet_v2(pretrained=True)
model.eval()

# ImageNetのクラスラベルを取得
LABELS_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
response = requests.get(LABELS_URL)
imagenet_labels = {int(key): value[1] for key, value in json.loads(response.text).items()}

# 画像の前処理手順を定義
preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
])

# 画像を読み込み，前処理する関数
def load_and_preprocess_image_pytorch(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t,0)
    return batch_t, img

# 画像を分類し，結果を表示する関数
def classify_image_pytorch(img_path):
    batch_t, original_img = load_and_preprocess_image_pytorch(img_path)

    with torch.no_grad():
        out = model(batch_t)
    
    probabilities = torch.nn.functional.softmax(out[0],dim=0)

    top3_prob, top3_catid = torch.topk(probabilities, 3)

    plt.imshow(original_img)
    plt.axis('off')
    plt.show()

    print("PyTorch 予測結果:")
    for i in range(top3_prob.size(0)):
        class_id = top3_catid[i].item()
        label = imagenet_labels.get(class_id,"Unknown")
        prob = top3_prob[i].item()
        print(f"{i+1}: {label} ({prob*100:.2f}%)")

image_file_path_pytorch = 'kiri.png'
classify_image_pytorch(image_file_path_pytorch)