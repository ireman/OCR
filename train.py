# Imports
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import os
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split)
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Cuneiform_Dataset import CYRUS

# param and defines:
torch.manual_seed(17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
num_epochs = 10
learning_rate = 0.001
batch_size = 128


def create_dict_of_sign(path='path_to_label_names.cdv'):
    df = pd.read_csv(path, keep_default_na=False, encoding='utf-8')
    label_list = df['label'].tolist()
    sign_dict = {label_list[i]: i for i in range(0, len(label_list))}
    return sign_dict


def dataset_from_csv(annotation_path='image_name_label_1.csv',
                     labels_name_file='filtered_label_from_Avital_ver3.csv'):
    sign_dict = create_dict_of_sign(labels_name_file)
    tablets_annotation = pd.read_csv(annotation_path, keep_default_na=False, encoding='utf-8')
    # df = pd.read_csv('/content/image_name_label.csv', keep_default_na=False,encoding='utf-8')
    train_data, test_data = train_test_split(tablets_annotation, test_size=0.36)
    # train.to_csv(path_or_buf='img_name_label_train.csv', index=None, header=True, encoding='utf-8')
    # test.to_csv(path_or_buf='img_name_label_test.csv', index=None, header=True, encoding='utf-8')
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomAffine(3), transforms.RandomAffine(4)])

    valid_transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = CYRUS(
        train_data,
        transform=train_transform,  # transforms.ToTensor(), transforms.GaussianBlur()
        sign_to_label=sign_dict
    )
    test_dataset = CYRUS(
        test_data,
        transform=valid_transform,  # transforms.ToTensor(), transforms.GaussianBlur()
        sign_to_label=sign_dict
    )
    val_len = int(0.4 * len(test_dataset))
    test_len = int(len(test_dataset) - val_len)
    val_ds, test_set = random_split(test_dataset, [val_len, test_len])
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(val_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=True)
    return train_loader, valid_loader, test_loader


def train_resnet18_classifier():
    model = torchvision.models.resnet18(pretrained=True)
    train_loader, valid_loader, test_loader = dataset_from_csv()
    num_batches_train = len(train_loader)
    num_batches_valid = len(valid_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_end_eval(num_epochs, model, criterion, optimizer, train_loader, valid_loader, test_loader, learning_rate,
                   num_batches_train, num_batches_valid)


def train_end_eval(epochs, model, criterion, optimizer, train_loader, valid_load, test_load, lr, num_batches_train,
                   num_batches_valid):
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []

    for epoch in range(epochs):
        correct = 0.0
        total = 0.0
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # for i, data in enumerate(train_load, 0):
            # inputs, labels = data
            # inputs = inputs.view(-1, 3*32*32).requires_grad_().to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('i = %d '% i)
        running_loss /= num_batches_train
        train_loss.append(running_loss)
        # print('total = %d' % total)
        train_acc = float(100 * correct / total)
        train_accuracy.append(train_acc)
        correct = 0.0
        total = 0.0
        valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_idx1, (images, labels) in enumerate(valid_load):
                # images = images.view(-1, 3*32*32).to(device)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).sum()
        model.train()
        valid_loss /= num_batches_valid
        # print('valid total = %d' % total)
        accuracy = float(100 * correct / total)
        #print('Accuracy of the network on valid images: %.3f %%' % (accuracy))
        validation_loss.append(valid_loss)
        validation_accuracy.append(accuracy)
        print("Epoch %d : train loss = %.3f  valid loss = %.3f Train accuracy: %.3f validation accuracy: %.3f "
              % (epoch + 1, running_loss, valid_loss, train_acc, accuracy))
    plot_result(epochs, batch_size, train_loss,
                validation_loss, train_accuracy, validation_accuracy, lr)
    print('Finished Training')

    correct = 0.0
    total = 0.0
    with torch.no_grad():
        model.eval()
        for batch_idx2, (images, labels) in enumerate(test_load):
            # images = images.view(-1, 3*32*32).to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()

    print('Accuracy of the network on test images: %.3f %%' % (
            100 * correct / total))
    model.train()


def plot_result(num_epochs ,batch_size, train_loss, validation_loss, train_accuracy, validation_accuracy,
                lr):
    epochs = range(1, num_epochs + 1)
    plt.subplot(1, 2, 1)
    plt.suptitle(f'epochs = {num_epochs} batch size = {batch_size} learning rate = {lr}',
                 fontsize=12, color='b')
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss', fontsize=10, color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, validation_accuracy, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy', fontsize=10, color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()
    plt.close()
