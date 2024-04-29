from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(size=(32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(5)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(size=(32, 32))
])

