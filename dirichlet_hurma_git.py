import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResizeWithPadding:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        img = F.resize(img, self.size)
        delta_width = self.size[0] - img.size[0]
        delta_height = self.size[1] - img.size[1]
        padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
        return F.pad(img, padding, self.fill)

def get_transform():
    return transforms.Compose([
        ResizeWithPadding((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load the datasets
dataset = torchvision.datasets.ImageFolder("path", transform=get_transform())
dataset_size = len(dataset)

# Define the K-Fold Cross Validator
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True)

def initialize_model(model_name):
    if model_name == 'VGG16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 9)
    elif model_name == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 9)
    elif model_name == 'MobileNetV2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 9)
    elif model_name == 'DenseNet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 9)
    return model


class DirichletEnsemble:
    def __init__(self, models, alpha=1.0):
        self.models = models
        self.alpha = alpha
    
    def predict(self, x):
        self.models = [model.eval() for model in self.models]
        with torch.no_grad():
            predictions = torch.stack([model(x) for model in self.models])
        predictions = nn.functional.softmax(predictions, dim=-1)
        alpha_post = predictions + self.alpha
        dirichlet_means = alpha_post / alpha_post.sum(dim=-1, keepdim=True)
        return dirichlet_means.mean(dim=0)
    
def eval_Drichlet(model, val_loader):
    correct = 0
    total = 0
    val_running_loss = 0.0
    all_labels = []
    all_preds = []
    criterion = nn.CrossEntropyLoss()
    ensemble.models = [model.eval() for model in ensemble.models]
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = ensemble.predict(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    val_loss.append(val_running_loss / len(val_loader))
    val_accuracy.append(correct / total)

    accuracy = val_accuracy[-1]
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1, train_loss, val_loss, val_accuracy, all_labels, all_preds

def train_and_evaluate(model, train_loader, val_loader, num_epochs=35):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        val_loss.append(val_running_loss / len(val_loader))
        val_accuracy.append(correct / total)
    
    accuracy = val_accuracy[-1]
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1, train_loss, val_loss, val_accuracy, all_labels, all_preds



model_names = ['VGG16', 'ResNet18', 'MobileNetV2', 'DenseNet121']
results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for name in model_names}
dirichlet_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

fold_metrics = {name: {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'labels': [], 'preds': []} for name in model_names}
fold_metrics['DirichletEnsemble'] = {'labels': [], 'preds': []}

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')
    train_subsampler = Subset(dataset, train_ids)
    val_subsampler = Subset(dataset, val_ids)

    train_loader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=32, shuffle=False)

    models = [initialize_model(name) for name in model_names]
    for i, model in enumerate(models):
        model = model.to(device)
        accuracy, precision, recall, f1, train_loss, val_loss, val_accuracy, labels, preds = train_and_evaluate(model, train_loader, val_loader)
        results[model_names[i]]['accuracy'].append(accuracy)
        results[model_names[i]]['precision'].append(precision)
        results[model_names[i]]['recall'].append(recall)
        results[model_names[i]]['f1'].append(f1)
        fold_metrics[model_names[i]]['train_loss'].append(train_loss)
        fold_metrics[model_names[i]]['val_loss'].append(val_loss)
        fold_metrics[model_names[i]]['val_accuracy'].append(val_accuracy)
        fold_metrics[model_names[i]]['labels'].extend(labels)
        fold_metrics[model_names[i]]['preds'].extend(preds)
        print(f'{model_names[i]}: Fold {fold+1} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    ensemble = DirichletEnsemble(models)
    accuracy, precision, recall, f1, _, _, _, labels, preds = eval_Drichlet(ensemble, val_loader)
    dirichlet_results['accuracy'].append(accuracy)
    dirichlet_results['precision'].append(precision)
    dirichlet_results['recall'].append(recall)
    dirichlet_results['f1'].append(f1)
    fold_metrics['DirichletEnsemble']['labels'].extend(labels)
    fold_metrics['DirichletEnsemble']['preds'].extend(preds)
    print(f'DirichletEnsemble: Fold {fold+1} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('--------------------------------')

# Average results
for name in model_names:
    print(f'{name}: Average Accuracy: {np.mean(results[name]["accuracy"]):.4f}, '
          f'Average Precision: {np.mean(results[name]["precision"]):.4f}, '
          f'Average Recall: {np.mean(results[name]["recall"]):.4f}, '
          f'Average F1: {np.mean(results[name]["f1"]):.4f}')
    
print(f'DirichletEnsemble: Average Accuracy: {np.mean(dirichlet_results["accuracy"]):.4f}, '
      f'Average Precision: {np.mean(dirichlet_results["precision"]):.4f}, '
      f'Average Recall: {np.mean(dirichlet_results["recall"]):.4f}, '
      f'Average F1: {np.mean(dirichlet_results["f1"]):.4f}')
      


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Her fold için metrikleri saklamak üzere bir sözlük oluştur
metrics_data = {
    model_name: {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'labels': [],
        'preds': []
    } for model_name in model_names
}


def update_metrics(metrics_data, model_names, fold_metrics, labels, preds):
    for model_name in model_names:
        metrics_data[model_name]['train_loss'].append(fold_metrics[model_name]['train_loss'])
        metrics_data[model_name]['val_loss'].append(fold_metrics[model_name]['val_loss'])
        metrics_data[model_name]['val_accuracy'].append(fold_metrics[model_name]['val_accuracy'])
        metrics_data[model_name]['labels'].append(fold_metrics[model_name]['labels'])
        metrics_data[model_name]['preds'].append(fold_metrics[model_name]['preds'])
        
    metrics_data['DirichletEnsemble']['labels'].append(fold_metrics['DirichletEnsemble']['labels'])
    metrics_data['DirichletEnsemble']['preds'].append(fold_metrics['DirichletEnsemble']['preds'])


update_metrics(metrics_data, model_names, fold_metrics, labels, preds)


with open('metrics_data.json', 'w') as json_file:
    json.dump(metrics_data, json_file,cls=NpEncoder)

