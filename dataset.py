import torch
import glob
import os
import cv2
import util
from torch.utils.data import Dataset
dic = {
    "buildings": 0,
    "forest": 1,
    "glacier": 2,
    "mountain": 3,
    "sea": 4,
    "street": 5
}



class MNISTDataset(Dataset):
    def __init__(self, root="../MINIST/nature32/", isTrain=True):
        super().__init__()
        self.isTrain = isTrain
        self.data = []
        type = "train" if isTrain else "test"

        paths = glob.glob(os.path.join(root, type, "*", "*"))
        for img_path in paths:
            img_infos = img_path.rsplit("/", maxsplit=2)
            label = img_infos[-2]
            self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path, label = self.data[idx]
        img = cv2.imread(img_path)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.isTrain:
            img_vector = util.train_transforms(img)
        else:
            img_vector = util.test_transforms(img)
        one_hot = torch.tensor(dic[label])
        return img_vector, one_hot

if __name__ == '__main__':
    dataset = MNISTDataset()
    print(len(dataset))
    print(dataset[5000])