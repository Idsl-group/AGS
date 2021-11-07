from Architectures.ResNet18 import *
from Model import *
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchattacks
import torchvision.transforms as transforms
import os


# Settings
config = {
    'batch_size': 100,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': torch.optim.SGD,
    'optimkwargs': {'lr':0.1, 'weight_decay':5e-4, 'momentum':0.9},
    'scheduler': torch.optim.lr_scheduler.MultiStepLR,
    'schedulerkwargs': {'milestones': [60, 120, 160], 'gamma':0.2},
    'epochs': 200,
    'in_channels': 3,
    'num_classes': 100,
    'n_steps': 10,
    'kernel_size': 3,
    'experiment_name': "nominal-cifar100",
    'model_log': None,
    'defense_log': None,
    'attack': None,
    'thres': 0,
    'attackkwargs': None,
    'comments': None
}
config['experiment_path'] = str(os.getcwd()) + "/AGS/Experiment_logs/" + config['experiment_name'] + "/"
if not os.path.isdir(config['experiment_path']):
    os.mkdir(config['experiment_path'])


# Load dataset
# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])
train_data = torchvision.datasets.CIFAR100("../data", train=True, download=True, transform=train_transform)
test_data = torchvision.datasets.CIFAR100("../data", train=False, download=True, transform=test_transform)
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.8), len(test_data) - int(len(test_data)*0.8)])
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
baselines = []
for x, y in train_loader:
  x = x.cuda()
  mean = x.mean(0).cuda()
  for i in range(0, len(x)):
    x[i] = mean
baselines = x.to(config['device'])


resnet18 = ResNet18(in_channels=config['in_channels'], num_classes=config['num_classes'])
defense = Model(resnet18, config, config['experiment_name'])


defense.train(train_loader, test_loader)


fgsm = torchattacks.FGSM(copy.deepcopy(defense.model), eps=8/255)
pgd_linf = torchattacks.PGD(copy.deepcopy(defense.model), eps=8/255, alpha=0.01, steps=100, random_start=False)
autoattack = torchattacks.AutoAttack(copy.deepcopy(defense.model), norm='Linf', eps=8/255, version='standard', n_classes=config['num_classes'], verbose=False)
list_of_attacks = [None] #, fgsm, pgd_linf, autoattack]
defense.validate(val_loader, list_of_attacks=list_of_attacks)