import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from Helper import Helper
from Patching import Patching


class SaveTrainRandomDataResults(tf.keras.callbacks.Callback):
    def __init__(self, results_path):
        super().__init__()
        self.filepath = results_path / Path("train_log.csv")

    def on_train_begin(self, logs=None):
        # Create file + header at start
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy"])

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_accuracy])

        print(f"Train: Epoch {epoch}, Loss {train_loss}, Accuracy {train_accuracy}")

class SaveRandomNetwork(tf.keras.callbacks.Callback):
    def __init__(self, save_every, network_saving_path):
        super().__init__()
        self.save_every = save_every
        self.filepath = network_saving_path
        os.makedirs(self.filepath, exist_ok=True)

    def _save_model(self, epoch):
        path = f"model_epoch{epoch}.keras"
        model_path = self.filepath / Path(path)
        self.model.save(model_path)
        print(f"Saved model at epoch {epoch}")

    def on_train_begin(self, logs=None):
        # Save initial weights before any training
        self._save_model(0)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every == 0:
            self._save_model(epoch+1)

class SaveTestRandomDataResults(tf.keras.callbacks.Callback):
    def __init__(self, results_path, patch_size, test_data_dir, seed, num_classes, loss_function, testing):
        super(SaveTestRandomDataResults, self).__init__()
        self.results_path = results_path
        self.csv_path = results_path / Path("total_test_log.csv")
        self.sample_acc_path = results_path / Path("sample_test_log.csv")
        self.sample_freq_path = results_path / Path("sample_freq_log.csv")
        self.cm_path = results_path / Path("confusion_matrix.png")
        self.test_patching = Patching(patch_size, test_data_dir)
        self.test_dir = test_data_dir
        self.patch_size = patch_size
        self.seed = seed
        self.num_classes = num_classes
        self.loss_function = loss_function
        self.testing = testing

    def on_train_begin(self, logs=None):
        if self.testing != "not":
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["accuracy", "mean_iou", "accuracy_bg", "accuracy_class_1", "accuracy_class_2"])

            sample_csv_exists = os.path.isfile(self.sample_acc_path)
            with open(self.sample_acc_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                if not sample_csv_exists:
                    writer.writerow([
                        "sample", "loss", "accuracy", "mean_iou",
                        "accuracy_bg", "accuracy_class_1", "accuracy_class_2"
                    ])

            sample_freq_csv_exists = os.path.isfile(self.sample_freq_path)
            with open(self.sample_freq_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                if not sample_freq_csv_exists:
                    writer.writerow([
                        "sample", "loss",
                        "true0_pred0", "true0_pred1", "true0_pred2",
                        "true1_pred0", "true1_pred1", "true1_pred2",
                        "true2_pred0", "true2_pred1", "true2_pred2",
                    ])
    def intersection_union_per_patch(self, y_true, y_pred, num_classes):
        intersections = np.zeros(num_classes, dtype=np.int64)
        unions = np.zeros(num_classes, dtype=np.int64)

        for c in range(num_classes):
            y_true_c = (y_true == c)
            y_pred_c = (y_pred == c)

            intersections[c] = np.logical_and(y_true_c, y_pred_c).sum()
            unions[c] = np.logical_or(y_true_c, y_pred_c).sum()

        return intersections, unions

    def update_dict(self, sample_csv_dict, pred_dict, gt_dict, original_path, intersections, unions, patch_voxels, loss_value, correct, pred_np, gt_np):
        if original_path in sample_csv_dict:
            current_array = sample_csv_dict[original_path]
            total_intersection = current_array[0] + intersections
            total_unions = current_array[1] + unions
            total_voxels = current_array[2] + patch_voxels
            total_loss = current_array[3] + (loss_value * patch_voxels)
            total_correct = current_array[4] + correct

            update_array = [total_intersection, total_unions, total_voxels, total_loss, total_correct]
            sample_csv_dict[original_path] = update_array

            current_pred = pred_dict[original_path]
            current_pred = np.concatenate([current_pred, pred_np.flatten()])
            pred_dict[original_path] = current_pred

            current_gt = gt_dict[original_path]
            current_gt = np.concatenate([current_gt, gt_np.flatten()])
            gt_dict[original_path] = current_gt

        else:
            current_array = [intersections, unions, patch_voxels, loss_value, correct]
            current_pred = pred_np.flatten()
            current_gt = gt_np.flatten()

            sample_csv_dict[original_path] = current_array
            pred_dict[original_path] = current_pred
            gt_dict[original_path] = current_gt

        return sample_csv_dict, pred_dict, gt_dict

    def create_sample_output(self, sample_csv_dict, pred_dict, gt_dict):
        for key, value in sample_csv_dict.items():
            print("key", str(key))
            pred_dict_value = pred_dict[key]
            gt_dict_value = gt_dict[key]

            loss = value[3] / value[2]
            iou_per_class = value[0] / (value[1] + 1e-7)
            valid_classes = value[1] > 0
            mean_iou = np.mean(iou_per_class[valid_classes]) if np.any(valid_classes) else 0.0

            cm = confusion_matrix(gt_dict_value, pred_dict_value, labels=list(range(self.num_classes)))
            per_class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-7)
            with open(self.sample_acc_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([key, value[3]/value[2], value[4]/value[2], mean_iou, per_class_accuracy[0], per_class_accuracy[1], per_class_accuracy[2]])

            with open(self.sample_freq_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    key, loss,
                    cm[0, 0], cm[0, 1], cm[0, 2],
                    cm[1, 0], cm[1, 1], cm[1, 2],
                    cm[2, 0], cm[2, 1], cm[2, 2],
                ])

            key_number = re.search(r'\d+', key).group()
            cm_sample_path_string = "confusion_matrix_" + str(key_number) + ".png"
            cm_sample_path = self.results_path / Path(cm_sample_path_string)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")

            plt.savefig(cm_sample_path)
            plt.close()

    def test_test_dataset(self):
        total_loss = 0.0
        total_correct = 0
        total_voxels = 0
        all_preds = []
        all_gt = []

        total_intersections = np.zeros(self.num_classes)
        total_unions = np.zeros(self.num_classes)
        current_patch_position = [0, 0, 0]
        last_patch_position, total_patches = Helper.count_patches(self.test_dir, self.patch_size)
        sample_csv_dict = {}
        pred_dict = {}
        gt_dict = {}

        for current_patch_number in range(total_patches):

            test_dataset, current_patch_position = self.test_patching.create_patch_dataset(
                current_patch_number,
                current_patch_position,
                last_patch_position
            )

            test_dataset = self.test_patching.shuffle(test_dataset, self.seed, 1)
            for image_patch, gt_patch, original_path, label_path in test_dataset:
                original_path = original_path.numpy()[0].decode("utf-8")

                pred_logits = self.model(image_patch, training=False)
                loss_value = self.loss_function(gt_patch, pred_logits).numpy()

                pred = tf.argmax(pred_logits, axis=-1)
                pred_np = pred.numpy()
                gt_np = gt_patch.numpy()

                pred_np = np.squeeze(pred_np, axis=0)
                gt_np = np.squeeze(gt_np, axis=0)

                gt_np = np.squeeze(gt_np, axis=-1)

                all_preds.append(pred_np.flatten())
                all_gt.append(gt_np.flatten())

                correct = tf.reduce_sum(tf.cast(tf.equal(tf.cast(gt_np, pred_np.dtype), pred_np), tf.int64)).numpy()

                intersections, unions = self.intersection_union_per_patch(gt_np, pred_np, self.num_classes)
                total_intersections += intersections
                total_unions += unions
                patch_voxels = np.prod(pred_np.shape)
                total_loss += float(loss_value) * patch_voxels
                total_correct += int(correct)
                total_voxels += int(patch_voxels)

                sample_csv_dict, pred_dict, gt_dict = self.update_dict(sample_csv_dict, pred_dict, gt_dict, original_path, intersections, unions, patch_voxels, loss_value, correct, pred_np, gt_np)

        self.create_sample_output(sample_csv_dict, pred_dict, gt_dict)

        return

    def on_train_end(self, logs=None):
        if self.testing != "not":
            self.test_test_dataset()

            df = pd.read_csv(self.sample_freq_path)
            cm_total = np.array([
                [
                    df["true0_pred0"].sum(),
                    df["true0_pred1"].sum(),
                    df["true0_pred2"].sum(),
                ],
                [
                    df["true1_pred0"].sum(),
                    df["true1_pred1"].sum(),
                    df["true1_pred2"].sum(),
                ],
                [
                    df["true2_pred0"].sum(),
                    df["true2_pred1"].sum(),
                    df["true2_pred2"].sum(),
                ],
            ])

            sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Overall confusion matrix")
            plt.savefig(self.cm_path)
            plt.close()

            tp = np.diag(cm_total)
            fn = cm_total.sum(axis=1) - tp
            fp = cm_total.sum(axis=0) - tp

            test_accuracy = tp.sum() / cm_total.sum()
            per_class_acc = tp / (tp + fn)
            per_class_iou = tp / (tp + fp + fn)
            test_mean_iou = np.mean(per_class_iou)

            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([test_accuracy, test_mean_iou, per_class_acc[0], per_class_acc[1], per_class_acc[2]])

            print(f"Test: Accuracy {test_accuracy}, Mean IoU {test_mean_iou}, Accuracy Background {per_class_acc[0]}, Accuracy Class 1 {per_class_acc[1]}, Accuracy Class 2 {per_class_acc[2]}")

    def set_model(self, model):
        super().set_model(model)