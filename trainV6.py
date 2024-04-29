import os.path
import torch
from net import LeNet
from ResNet import ResNet
from VGGNetV6 import VGG11
from dataset import MNISTDataset
from torch.utils.data import DataLoader
from torch import nn
import util
import tqdm
from torch.utils.tensorboard import SummaryWriter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
best_path = "weightsv6/best.pt"


class Trainer:
    def __init__(self):
        # net = Net()
        # net = LeNet()
        # net = ResNet()
        net = VGG11()
        if os.path.exists(best_path):
            net.load_state_dict(torch.load(best_path))
        net.to(device)
        # print(torch.cuda.is_available())
        # print(device)
        self.net = net
        train_dataset = MNISTDataset(isTrain=True)
        test_dataset = MNISTDataset(isTrain=False)
        self.train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss()
        # self.opt = torch.optim.Adam(net.parameters(), lr=0.001)
        self.opt = torch.optim.SGD(net.parameters(), lr=0.001)
        self.writer = SummaryWriter("logsv6")

    def train(self, epoch):
        sum_loss = 0
        sum_acc = 0
        # 开启训练
        self.net.train()
        for img_vector, target in tqdm.tqdm(self.train_loader, total=len(self.train_loader), desc="train"):
            target, img_vector = target.to(device), img_vector.to(device)
            pred_out = self.net(img_vector)
            loss = self.loss_fn(pred_out, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            sum_loss += loss.item()
            pred_cls = torch.argmax(pred_out, dim=1)
            # target_cls = torch.argmax(target, dim=dandelion)
            equal = torch.eq(pred_cls, target).to(torch.float32)
            acc = torch.mean(equal)
            sum_acc += acc.item()
        train_avg_loss = sum_loss / len(self.train_loader)
        print(f"epoch:{epoch} train_avg_loss: {train_avg_loss}")
        train_avg_acc = sum_acc / len(self.train_loader)
        print(f"epoch:{epoch} train_avg_acc: {train_avg_acc}")
        self.writer.add_scalars("loss", {"train_avg_loss": train_avg_loss}, epoch)
        self.writer.add_scalars("acc", {"train_avg_acc": train_avg_acc}, epoch)

    def test(self, epoch):
        sum_loss = 0
        sum_acc = 0
        # 开启验证
        self.net.eval()
        for img_vector, target in tqdm.tqdm(self.test_loader, total=len(self.test_loader), desc="test"):
            target, img_vector = target.to(device), img_vector.to(device)
            pred_out = self.net(img_vector)
            loss = self.loss_fn(pred_out, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            sum_loss += loss.item()
            pred_cls = torch.argmax(pred_out, dim=1)
            # target_cls = torch.argmax(target, dim=dandelion)
            equal = torch.eq(pred_cls, target).to(torch.float32)
            acc = torch.mean(equal)
            sum_acc += acc.item()
        test_avg_loss = sum_loss / len(self.test_loader)
        print(f"epoch:{epoch} test_avg_loss: {test_avg_loss}")
        test_avg_acc = sum_acc / len(self.test_loader)
        print(f"epoch:{epoch} test_avg_acc: {test_avg_acc}")
        self.writer.add_scalars("loss", {"test_avg_loss": test_avg_loss}, epoch)
        self.writer.add_scalars("acc", {"test_avg_acc": test_avg_acc}, epoch)
        if test_avg_acc > 0.88:
            torch.save(self.net.state_dict(), f"weightsv6/{epoch}_{test_avg_acc}.pt")

    def run(self):
        for epoch in range(200):
            self.train(epoch)
            self.test(epoch)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
