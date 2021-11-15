from Architectures.CNN4_4 import *
from Model import *
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.transforms  import ToTensor
import copy
import torchattacks
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
    'epochs': 20,
    'in_channels': 1,
    'num_classes': 10,
    'n_steps': 10,
    'kernel_size': 3,
    'experiment_name': "nominal-mnist",
    'model_log': None,
    'attack': None,
    'thres': 0.5,
    'attackkwargs': {},
    'comments': None
}


config['experiment_path'] = str(os.getcwd()) + "/Experiment_logs/" + config['experiment_name'] + "/"
if not os.path.isdir(config['experiment_path']):
    os.mkdir(config['experiment_path'])


# Load dataset
train_data = torchvision.datasets.MNIST("../data", train=True, download=True, transform=ToTensor())
test_data = torchvision.datasets.MNIST("../data", train=False, download=True, transform=ToTensor())
test_data, val_data = torch.utils.data.random_split(test_data, [int(len(test_data)*0.8), len(test_data) - int(len(test_data)*0.8)])
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True, num_workers=0)
for x, y in train_loader:
  baselines = torch.zeros_like(x).cuda()
  break


cnn = CNN4_4(in_channels=config['in_channels'], num_classes=config['num_classes'])
defense = Model(cnn, config, config['experiment_name'])


defense.train(val_loader, test_loader)


fgsm = torchattacks.FGSM(copy.deepcopy(defense.model), eps=0.3)
pgd_linf = torchattacks.PGD(copy.deepcopy(defense.model), eps=0.3, alpha=0.1, steps=100, random_start=False)
autoattack = torchattacks.AutoAttack(copy.deepcopy(defense.model), norm='Linf', eps=0.3, version='standard', n_classes=config['num_classes'], verbose=False)
list_of_attacks = [None] #, fgsm, pgd_linf, autoattack]
defense.validate(val_loader, list_of_attacks=list_of_attacks)