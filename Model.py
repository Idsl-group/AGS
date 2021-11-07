import torch
import torch.nn as nn
import datetime
from prettytable import PrettyTable
import copy

class Model:
    def __init__(self, defense, config, defense_name=None):
        """
            Adapt epoch(), train(), validation_epoch() and validate() for required experiment.

            Params:
                model: Should be a class object that inherits the nn.Module with a properly defined forward function.
                      Always try to use model as described in other benchmark.
                criterion: any function whose gradient can be tracked will work here. nn.Loss objects work better
                optimizer: nn.optim.optimizer object
                scheduler: nn.optim.scheduler object
                device: 'cpu' or 'cuda'
                optimkwargs: dictionary of optimizer kwargs
                schedulerkwargs: dictionary of scheduler kwargs
        """
        self.config = config
        self.device = self.config['device']
        self.criterion = self.config['criterion']
        self.model = defense
        self.defense_name = defense_name

        # Set up log path
        defense_log = self.defense_name + "_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")[:19]
        self.log_path = self.config['experiment_path'] + defense_log + ".txt"
        self.defense_path = self.config['experiment_path'] + defense_log + ".pth"
        with open(self.log_path, 'a') as fp:
            new_log = "New log: " + str(datetime.datetime.now())
            fp.write(f"{new_log}\n")
            pass

        # Set up optimizer and scheduler for training
        self.optimizer = self.config['optimizer'](self.model.parameters(), **self.config['optimkwargs'])
        if self.config['scheduler']:
            self.scheduler = self.config['scheduler'](self.optimizer, **self.config['schedulerkwargs'])
        else:
            self.scheduler = None
        self.best_model = copy.deepcopy(self.model)

        # Data parallelism
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model.cuda())

    def logit_transformation(self, yp):
        return yp.max(dim=1)[1]

    def epoch(self, loader, train=False, attack=None, thres=0):
        total_err, total_loss = 0., 0.
        for x,y in loader:
            x, y = x.to(self.device), y.to(self.device)
            if attack:
                rand = torch.rand(1)
                if rand > thres:
                    x = attack(x, y).type(torch.FloatTensor).cuda()
            yp = self.model(x)
            loss = self.criterion(yp, y)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            yp = self.logit_transformation(yp)
            total_err += (yp != y).sum().item()
            total_loss += loss.item() * x.shape[0]
            torch.cuda.empty_cache()
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def train(self, train_loader, test_loader, ):
        acc = 0
        train_info = ['TRAINING WITH']
        train_info.append(" MODEL")
        train_info.append(self.model)
        train_info.append(" MODEL")
        for elem in self.config:
            train_info.append("{}: {}".format(elem, self.config[elem]))
        with open(self.log_path, 'a') as fp:
            for item in train_info:
                print(item)
                fp.write(f"{item}\n")
            pass
        if self.config['attack']:
            attack = self.config['attack'](copy.deepcopy(self.model), **self.config['attackkwargs'])
        else:
            attack=None
        for epoch in range(0, self.config['epochs']):
          now = datetime.datetime.now().timestamp()
          train_err, train_loss = self.epoch(train_loader, train=True, attack=attack, thres=self.config['thres'])
          test_err, test_loss = self.epoch(test_loader, attack=attack, thres=self.config['thres'])
          if self.scheduler:
              self.scheduler.step()
          torch.cuda.empty_cache()
          after = datetime.datetime.now().timestamp()
          epoch_info = ["{}  Train_accuracy: {:.6f} Train_loss:  {:.6f}  Test_accuracy: {:.6f}  Test_loss: {:.6f}  Time: {:.2f} s".format(epoch, 1-train_err, train_loss, 1-test_err, test_loss, after-now)]
          new_acc = 1 - test_err
          if new_acc > acc:
              if torch.cuda.device_count() > 1:
                torch.save(self.model.module.state_dict(), self.defense_path)
              else:
                  torch.save(self.model.state_dict(), self.defense_path)
              acc = new_acc
              epoch_info.append("   Model saved at: {}".format(self.defense_path))
          with open(self.log_path, 'a') as fp:
              for item in [epoch_info]:
                  print(item)
                  fp.write(f"{item}\n")
              pass

    def validation_epoch(self, loader, attack, img_lim=-1):
        total_err = 0.
        i = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            if attack:
                x = attack(x, y).type(torch.FloatTensor).cuda()
            yp = self.model(x)
            yp = self.logit_transformation(yp)
            total_err += (yp != y).sum().item()
            i += len(x)
            if i >= img_lim and img_lim > 0:
                return total_err / (i)
        return total_err / len(loader.dataset)

    def validate(self, val_loader, list_of_attacks = [], img_lim=-1):
        val_info = ['\n']
        val_info.append("VALIDATING WITH:")
        val_info.append(' MODEL')
        val_info.append(self.model)
        val_info.append(' MODEL')
        val_info.append('Validation info:')
        for elem in self.config:
            val_info.append("{}: {}".format(elem, self.config[elem]))

        with open(self.log_path, 'a') as fp:
            for item in val_info:
                print(item)
                fp.write(f"{item}\n")
            pass
        for attack in list_of_attacks:
            now = datetime.datetime.now().timestamp()
            val_epoch_info = ["Adversarial accuracy against {}".format(attack)]
            test_err = self.validation_epoch(val_loader, attack=attack, img_lim=img_lim)
            after = datetime.datetime.now().timestamp()
            val_epoch_info.append("    Acc: {} in {} s".format(round((1 - test_err), 4), after - now))
            with open(self.log_path, 'a') as fp:
                for item in val_epoch_info:
                    print(item)
                    fp.write(f"{item}\n")
                pass

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
