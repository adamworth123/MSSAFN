import os
import time
import random
import numpy as np
import torch
from torch import nn, optim
from pt_dcn_models import PT_DCN
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import matplotlib
from torch.nn.utils import clip_grad_norm_

matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()
print("Is CUDA available:", torch.cuda.is_available())

seed = 23
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

MRI_DATA_DIR = r'/root/MRI'
FDG_DATA_DIR = r'/root/FDG'
MODEL_PATH = r'/root/model'
RESULTS_PATH = r'/root/result'

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
LR = 1e-4
EPOCH = 300
WORKERS = 10
WEIGHT_DECAY = 1e-3
patience = 35


class MRI_FDG_Dataset(Dataset):
    def __init__(self, mri_data_path, fdg_data_path, split='train'):
        self.mri_X = np.load(os.path.join(mri_data_path, f'X_{split}.npy'))
        self.mri_y = np.load(os.path.join(mri_data_path, f'y_{split}.npy'))
        self.fdg_X = np.load(os.path.join(fdg_data_path, f'X_{split}.npy'))
        self.fdg_y = np.load(os.path.join(fdg_data_path, f'y_{split}.npy'))
        print(f"\nChecking {split} data dimensions:")
        print(f"MRI data shape: {self.mri_X.shape}")
        print(f"FDG data shape: {self.fdg_X.shape}")

        assert np.array_equal(self.mri_y, self.fdg_y), "MRI and FDG labels do not match"
        self.labels = self.mri_y
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mri_img = self.mri_X[idx]
        fdg_img = self.fdg_X[idx]
        image = np.stack((mri_img, fdg_img), axis=0)
        image = torch.from_numpy(image).float()
        label = torch.tensor(self.labels[idx]).long()
        return image, label


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.001, best_model_path=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.Inf
        self.best_model_path = best_model_path
        self.best_model_filename = None

    def __call__(self, val_metric, val_loss, val_acc, model):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, val_loss, val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, val_loss, val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, val_loss, val_acc, model):
        if self.verbose:
            print(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_metric:.6f}). Saving model ...')

        auc_str = f"{val_metric:.4f}".replace('.', '_')
        loss_str = f"{val_loss:.4f}".replace('.', '_')
        acc_str = f"{val_acc:.4f}".replace('.', '_')

        model_filename = f'best_model_AUC_{auc_str}_Loss_{loss_str}_Acc_{acc_str}.pth'
        model_path = os.path.join(self.best_model_path, model_filename)

        torch.save(model.state_dict(), model_path)
        self.val_auc_max = val_metric
        self.best_model_filename = model_filename


def plot_and_save_curves(train_metrics, val_metrics, metric_name, results_path):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics, label=f"Train {metric_name}")
    plt.plot(epochs, val_metrics, label=f"Val {metric_name}")
    plt.title(f"{metric_name} Curve")
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()

    save_path = os.path.join(results_path, f"{metric_name.lower()}_curve.png")
    print(f"Saving curve to {save_path}")
    plt.savefig(save_path)
    plt.close()


def plot_roc_pr_curves(fpr, tpr, scores, labels, auc_score, results_path, split):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve - {split}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, f"roc_curve_{split}.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {split}")
    plt.savefig(os.path.join(results_path, f"pr_curve_{split}.png"))
    plt.close()


def save_confusion_matrix(cm, results_path, split):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {split}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(results_path, f"confusion_matrix_{split}.png"))
    plt.close()


def compute_metrics(labels_list, scores_list, threshold=0.5):
    preds = (np.array(scores_list) >= threshold).astype(int)

    precision = precision_score(labels_list, preds, zero_division=0)
    recall = recall_score(labels_list, preds, zero_division=0)
    f1 = f1_score(labels_list, preds, zero_division=0)

    cm = confusion_matrix(labels_list, preds)

    if len(np.unique(labels_list)) > 1:
        fpr, tpr, _ = roc_curve(labels_list, scores_list)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None
        fpr, tpr = None, None

    if (cm.ravel().size == 4):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return precision, recall, f1, roc_auc, preds, fpr, tpr, specificity


def train_model():
    logging.basicConfig(
        filename=os.path.join(RESULTS_PATH, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    all_metrics = []

    print("\nStarting single training run")
    logging.info("Starting single training run")

    dataset_train = MRI_FDG_Dataset(MRI_DATA_DIR, FDG_DATA_DIR, split='train')
    dataset_valid = MRI_FDG_Dataset(MRI_DATA_DIR, FDG_DATA_DIR, split='val')
    dataset_test = MRI_FDG_Dataset(MRI_DATA_DIR, FDG_DATA_DIR, split='test')

    data_loader_train = DataLoader(dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    data_loader_valid = DataLoader(dataset_valid, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    data_loader_test = DataLoader(dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model = PT_DCN().cuda() if cuda else PT_DCN()

    labels_train = dataset_train.labels
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train), y=labels_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    if cuda:
        class_weights = class_weights.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda() if cuda else nn.CrossEntropyLoss(weight=class_weights)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    early_stopping = EarlyStopping(patience=patience, verbose=True, best_model_path=MODEL_PATH)
    writer = SummaryWriter(log_dir=os.path.join(RESULTS_PATH, 'logs'))

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(EPOCH):
        start_time = time.time()

        model.train()
        total_samples = 0
        running_loss = 0.0
        running_corrects = 0
        train_labels_list = []
        train_scores_list = []

        train_loader = tqdm(data_loader_train, desc=f"Epoch [{epoch + 1}/{EPOCH}] Training", leave=False)
        for images, labels in train_loader:
            labels = labels.cuda() if cuda else labels
            images = images.cuda() if cuda else images
            optimizer.zero_grad()
            mri_images = images[:, 0, :, :, :].unsqueeze(1)
            pet_images = images[:, 1, :, :, :].unsqueeze(1)
            outputs = model(mri_images, pet_images)
            loss = criterion(outputs, labels)
            loss.backward()
            max_norm = 1.0
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            running_corrects += (predicted == labels).sum().item()
            total_samples += batch_size

            probs = F.softmax(outputs, dim=1)
            scores = probs[:, 1].detach().cpu().numpy()
            train_labels_list.extend(labels.cpu().numpy())
            train_scores_list.extend(scores)

            avg_loss = running_loss / total_samples
            avg_acc = running_corrects / total_samples
            train_loader.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples
        train_precision, train_recall, train_f1, roc_auc_train, _, fpr_train, tpr_train, train_specificity = compute_metrics(
            train_labels_list, train_scores_list)

        model.eval()
        val_labels_list = []
        val_scores_list = []
        val_running_loss = 0.0
        val_running_corrects = 0
        total_val_samples = 0

        with torch.no_grad():
            val_loader = tqdm(data_loader_valid, desc=f"Epoch [{epoch + 1}/{EPOCH}] Validation", leave=False)
            for images, labels in val_loader:
                labels = labels.cuda() if cuda else labels
                images = images.cuda() if cuda else images
                mri_images = images[:, 0, :, :, :].unsqueeze(1)
                pet_images = images[:, 1, :, :, :].unsqueeze(1)
                outputs = model(mri_images, pet_images)
                loss = criterion(outputs, labels)
                batch_size = labels.size(0)
                val_running_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                val_running_corrects += (predicted == labels).sum().item()
                total_val_samples += batch_size

                probs = F.softmax(outputs, dim=1)
                scores = probs[:, 1].detach().cpu().numpy()
                val_labels_list.extend(labels.cpu().numpy())
                val_scores_list.extend(scores)

                avg_val_loss = val_running_loss / total_val_samples
                avg_val_acc = val_running_corrects / total_val_samples
                val_loader.set_postfix(loss=f"{avg_val_loss:.4f}", acc=f"{avg_val_acc:.4f}")

        val_loss = val_running_loss / total_val_samples
        val_acc = val_running_corrects / total_val_samples
        val_precision, val_recall, val_f1, roc_auc_val, _, fpr_val, tpr_val, val_specificity = compute_metrics(
            val_labels_list, val_scores_list)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch [{epoch + 1}/{EPOCH}] Metrics:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Train AUC: {roc_auc_train:.4f}" if roc_auc_train else "Train AUC: N/A", end=' | ')
        print(f"Val AUC: {roc_auc_val:.4f}" if roc_auc_val else "Val AUC: N/A")
        print(f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Train Specificity: {train_specificity:.4f} | Val Specificity: {val_specificity:.4f}")
        print(f"Train F1 Score: {train_f1:.4f} | Val F1 Score: {val_f1:.4f}")
        print(f"Time: {time.time() - start_time:.2f} sec")
        print(f"Current learning rate: {current_lr:.8f}")

        log_message = (
            f"\nEpoch [{epoch + 1}/{EPOCH}] Metrics:\n"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n"
            f"Train AUC: {roc_auc_train:.4f if roc_auc_train else 'N/A'} | Val AUC: {roc_auc_val:.4f if roc_auc_val else 'N/A'}\n"
            f"Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}\n"
            f"Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}\n"
            f"Train Specificity: {train_specificity:.4f} | Val Specificity: {val_specificity:.4f}\n"
            f"Train F1 Score: {train_f1:.4f} | Val F1 Score: {val_f1:.4f}\n"
            f"Time: {time.time() - start_time:.2f} sec\n"
            f"Current learning rate: {current_lr:.8f}"
        )
        logging.info(log_message)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUC/train', roc_auc_train if roc_auc_train else 0, epoch)
        writer.add_scalar('AUC/val', roc_auc_val if roc_auc_val else 0, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('Specificity/train', train_specificity, epoch)
        writer.add_scalar('Specificity/val', val_specificity, epoch)
        writer.add_scalar('F1_Score/train', train_f1, epoch)
        writer.add_scalar('F1_Score/val', val_f1, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        metrics_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_acc, 'val_acc': val_acc,
            'train_precision': train_precision, 'val_precision': val_precision,
            'train_recall': train_recall, 'val_recall': val_recall,
            'train_specificity': train_specificity, 'val_specificity': val_specificity,
            'train_f1': train_f1, 'val_f1': val_f1,
            'train_auc': roc_auc_train if roc_auc_train else np.nan,
            'val_auc': roc_auc_val if roc_auc_val else np.nan,
        }
        all_metrics.append(metrics_dict)

        early_stopping(roc_auc_val if roc_auc_val is not None else 0, val_loss, val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            logging.info("Early stopping")
            break

    writer.close()

    plot_and_save_curves(train_losses, val_losses, "Loss", RESULTS_PATH)
    plot_and_save_curves(train_accs, val_accs, "Accuracy", RESULTS_PATH)
    plot_and_save_curves(train_precisions, val_precisions, "Precision", RESULTS_PATH)
    plot_and_save_curves(train_recalls, val_recalls, "Recall", RESULTS_PATH)
    plot_and_save_curves(train_f1s, val_f1s, "F1", RESULTS_PATH)

    best_model_path = os.path.join(MODEL_PATH,
                                   early_stopping.best_model_filename) if early_stopping.best_model_filename else None
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
    else:
        print("Best model not found. Using the last state of the model for testing.")
        model.eval()

    test_labels_list = []
    test_scores_list = []
    with torch.no_grad():
        test_loader = tqdm(data_loader_test, desc="Testing")
        for images, labels in test_loader:
            labels = labels.cuda() if cuda else labels
            images = images.cuda() if cuda else images
            mri_images = images[:, 0, :, :, :].unsqueeze(1)
            pet_images = images[:, 1, :, :, :].unsqueeze(1)
            outputs = model(mri_images, pet_images)
            probs = F.softmax(outputs, dim=1)
            scores = probs[:, 1].detach().cpu().numpy()
            test_labels_list.extend(labels.cpu().numpy())
            test_scores_list.extend(scores)

    test_precision, test_recall, test_f1, roc_auc_test, test_preds, fpr_test, tpr_test, test_specificity = compute_metrics(
        test_labels_list, test_scores_list)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels_list))

    print("\nFinal Test Metrics:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {roc_auc_test:.4f}" if roc_auc_test else "AUC: N/A")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall (Sensitivity): {test_recall:.4f}")
    print(f"Specificity: {test_specificity:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    logging.info("\nFinal Test Metrics:")
    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info(f"AUC: {roc_auc_test:.4f}" if roc_auc_test else "AUC: N/A")
    logging.info(f"Precision: {test_precision:.4f}")
    logging.info(f"Recall (Sensitivity): {test_recall:.4f}")
    logging.info(f"Specificity: {test_specificity:.4f}")
    logging.info(f"F1 Score: {test_f1:.4f}")

    if roc_auc_test is not None:
        plot_roc_pr_curves(fpr_test, tpr_test, test_scores_list, test_labels_list, roc_auc_test, RESULTS_PATH, "test")

    cm = confusion_matrix(test_labels_list, test_preds)
    print(f"Confusion Matrix:\n{cm}")
    logging.info(f"Confusion Matrix:\n{cm}")
    save_confusion_matrix(cm, RESULTS_PATH, "test")

    metrics_df = pd.DataFrame(all_metrics)
    csv_path = os.path.join(RESULTS_PATH, 'training_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"All epoch metrics saved to {csv_path}")
    logging.info(f"All epoch metrics saved to {csv_path}")


if __name__ == '__main__':
    train_model()