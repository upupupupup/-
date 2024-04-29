import glob
import json
import os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
import tqdm
from torchvision.models import MobileNet_V2_Weights
from ResNetV11 import ResNet

test_transforms = transforms.Compose([
    # 将H W C 转为 C H W
    transforms.ToTensor(),
    # 尺寸统一
    # transforms.Resize((32, 32), antialias=True)
])
BASE_PATH = r'../MINIST/nature32/train'


class SearchImage:
    def __init__(self):
        self.model = self.load_model()
        self.db_names, self.db_feats = self.load_db_feat()

    def load_model(self):
        # 创建网络加载参数
        model = ResNet()
        state_dict = torch.load('./weightsv11/89_0.9653333206971486.pt')
        model.load_state_dict(state_dict)

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=6)
        )

        # 验证模式
        model.eval()
        return model

    def pre_process(self, img_path):
        # 预处理图片
        # H W C
        img = cv2.imread(img_path)
        # C H W
        img = test_transforms(img)
        # N C H W 升维
        img = torch.unsqueeze(img, dim=0)
        return img

    def extract_features(self, img_path):
        # 提取特征
        img = self.pre_process(img_path)
        x = self.model.forward_features(img)
        # 根据需要处理x以获取特征向量
        # 例如，使用adaptive_avg_pool2d来获得特定尺寸的特征图
        x = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))
        feat = torch.flatten(x, 1)[0].detach().numpy().astype(np.float64)
        return feat

    def init_img_features(self):
        # 存储特征
        img_paths = glob.glob(os.path.join(BASE_PATH, "*", "*"))
        with open("features32.txt", "w", encoding="utf-8") as file:
            for img_path in tqdm.tqdm(img_paths, total=len(img_paths)):
                # 获取完整的相对路径（包括子文件夹）
                rel_path = os.path.relpath(img_path, BASE_PATH)
                # 保存rel_path，feat
                feat = self.extract_features(img_path)
                feat = list(feat)
                # python对象转化为json字符串
                feat_str = json.dumps(feat)
                file.write(rel_path + "|" + feat_str + "\n")

    def load_db_feat(self):
        # 加载已存储的特征
        db_names, db_feats = [], []
        with open("features32.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                rel_path, feat_str = line.split("|")
                # 将json字符串转化为python对象
                feat = json.loads(feat_str)
                db_names.append(rel_path)  # 保存相对路径
                db_feats.append(feat)
        # 转化为np数组，加快处理速度
        db_feats = np.array(db_feats)
        # object是保持原有类型
        db_names = np.array(db_names, dtype=object)
        return db_names, db_feats

    def calc_similarity(self, img_path):
        # 计算相似度
        # 提取当前图片特征
        img_feat = self.extract_features(img_path)
        # 获取已存储的特征
        db_feats = self.db_feats
        db_names = self.db_names
        # 计算距离
        dist = np.linalg.norm(db_feats - img_feat, axis=1)
        # 排序
        dist_name = np.column_stack((dist, db_names))
        sort_idx = np.argsort(dist_name[:, 0])[:20]
        sort_idx_name = dist_name[sort_idx]
        # print(sort_idx_name)
        img_names = sort_idx_name[:, 1]
        # 可视化
        # self.visualize_img(img_names)
        self.path_show(img_names)

    def visualize_img(self, img_names):
        for img_name in img_names:
            img_path = os.path.join(BASE_PATH, img_name)  # img_name 是相对路径
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imshow("img", img)
                cv2.waitKey(500)
            else:
                print(f"Failed to load image: {img_name}")

    def path_show(self, img_names):
        for img_name in img_names:
            print(f"{img_name}")


if __name__ == '__main__':
    search = SearchImage()
    img_path = r"../MINIST/nature32/train/mountain/301.jpg"
    # search.init_img_features()
    search.calc_similarity(img_path)