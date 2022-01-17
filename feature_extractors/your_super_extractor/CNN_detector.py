import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy


class CNNEarDetector:

    def __init__(self, model, num_classes, extract_features=True):
        if model == "resnet":
            self.model = torchvision.models.resnet50(pretrained=True)
            if extract_features:
                for param in model.parameters():
                    param.requires_grad = False
            nfeatures = self.model.fc.in_features
            self.model.fc = nn.Linear(nfeatures, num_classes)
            self.input_size = 224
        elif model == "alexnet":
            self.model = torchvision.models.alexnet(pretrained=True)
            if extract_features:
                for param in model.parameters():
                    param.requires_grad = False
            nfeatures = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(nfeatures, num_classes)
            self.input_size = 224

        self.extract_features = extract_features

        self.optimizer = None

    def set_otpimizer(self, learning_rate=0.001, weight_decay=0):
        # construct an optimizer
        if self.extract_features:
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()

        self.optimizer = torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)

    def train(self, dataloader, criterion, num_epochs=50, learning_rate=0.001, weight_decay=0, since=time.time()):
        # train on the GPU or on the CPU, if a GPU is not available
        device = None
        if torch.cuda.is_available():
            print("Training model on GPUs")
            device = torch.device('cuda')
        else:
            print("Training model on CPUs")
            device = torch.device('cpu')

        self.model.to(device)

        # Initialize optimizer if it doesn't already exist
        if self.optimizer is None:
            self.set_otpimizer(learning_rate, weight_decay)

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #print(labels)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        return val_acc_history

    def save(self, root, filename):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   root+"/feature_extractors/your_super_extractor/"+filename+".pt")

    def load(self, root, filename):
        if torch.cuda.is_available():
            checkpoint = torch.load(root+"/feature_extractors/your_super_extractor/"+filename+".pt")
        else:
            checkpoint = torch.load(root +"/feature_extractors/your_super_extractor/" + filename + ".pt", map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint['model_state_dict'])

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.001)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if torch.cuda.is_available():
            self.model.to('cuda')

    def __call__(self, *args):
        return self.model(*args)

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()
