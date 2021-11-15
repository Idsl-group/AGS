from Architectures.ResNet18 import *
from Model import *
import torchattacks
from torch.utils.data.dataloader import DataLoader
import torchvision
import  torchvision.transforms  as T
import os

# Settings
config = {
    'batch_size': 50,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'criterion': nn.CrossEntropyLoss(),
    'optimizer': torch.optim.Adam,
    'optimkwargs': {},
    'scheduler': None,
    'schedulerkwargs': {},
    'epochs': 0,
    'in_channels': 3,
    'num_classes': 100,
    'n_steps': 10,
    'kernel_size': 3,
    'experiment_name': "ags-cifar100",
    'model_log': None,
    'attack': None,
    'thres': 0,
    'attackkwargs': {},
    'comments': None
}
config['experiment_path'] = str(os.getcwd()) + "/Experiment_logs/" + config['experiment_name'] + "/"
if not os.path.isdir(config['experiment_path']):
    os.mkdir(config['experiment_path'])

# Load dataset
normalize = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = T.Compose([])
train_transform.transforms.append(T.RandomCrop(32, padding=4))
train_transform.transforms.append(T.RandomHorizontalFlip())
train_transform.transforms.append(T.ToTensor())
train_transform.transforms.append(normalize)
test_transform = T.Compose([
    T.ToTensor(),
    normalize])
train_data = torchvision.datasets.CIFAR100("../data", train=True, download=True, transform=train_transform)
test_data = torchvision.datasets.CIFAR100("../data", train=False, download=True, transform=test_transform)
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.5), len(test_data) - int(len(test_data)*0.5)])
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

resnet = ResNet18(in_channels=config['in_channels'], num_classes=config['num_classes'])
ags_cnn = AGS_ResNet18(resnet, baselines=baselines, n_steps=config['n_steps'], kernel_size=config['kernel_size'], log_path=config['model_log'])
defense = Model(ags_cnn, config, config['experiment_name'])

fgsm = torchattacks.FGSM(copy.deepcopy(defense.model), eps=8/255)
pgd_linf = torchattacks.PGD(copy.deepcopy(defense.model), eps=8/255, alpha=0.1, steps=100, random_start=False)
autoattack = torchattacks.AutoAttack(copy.deepcopy(defense.model), norm='Linf', eps=8/255, version='standard', n_classes=config['num_classes'], verbose=False)
list_of_attacks =  [None, fgsm, pgd_linf, autoattack]
defense.validate(val_loader, list_of_attacks=list_of_attacks)