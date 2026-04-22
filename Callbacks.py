from pathlib import Path
import csv
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Patching import Patching


class SaveTrainDataResults(tf.keras.callbacks.Callback):
    def __init__(self, patch, results_path):
        super().__init__()
        self.patch = patch
        self.filepath = results_path / Path("train_log" + str(patch) + ".csv")

    def on_train_begin(self, logs=None):
        # Create file + header at start
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["patch", "epoch", "loss", "accuracy"])

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        train_accuracy = logs.get('accuracy')

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.patch, epoch, train_loss, train_accuracy])

        print(f"Train: Patch {self.patch}, Epoch {epoch}, Loss {train_loss}, Accuracy {train_accuracy}")

class SaveNetwork(tf.keras.callbacks.Callback):
    def __init__(self, save_every, patch, network_saving_path):
        super().__init__()
        self.save_every = save_every
        self.patch = patch
        self.filepath = network_saving_path / Path("patch" + str(patch))
        os.makedirs(self.filepath, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every == 0:
            path = "model_epoch{epoch}.keras".format(epoch=epoch + 1)
            model_path = self.filepath / Path(path)
            self.model.save(model_path)
            print(f"Saved model at patch {self.patch} and epoch {epoch+1}")

class SaveTestDataResults(tf.keras.callbacks.Callback):
    def __init__(self, patch, results_path, patch_size, test_data_dir, seed, num_classes, total_patches, last_patch_position, loss_function, testing, confusion_matrix):
        super(SaveTestDataResults, self).__init__()
        self.patch = patch
        self.csv_path = results_path / Path("test_log" + str(patch) + ".csv")
        self.cm_path = results_path / Path("confusion_matrix_pretty" + str(patch) + ".png")
        self.test_patching = Patching(patch_size, test_data_dir)
        self.seed = seed
        self.num_classes = num_classes
        self.total_patches = total_patches
        self.last_patch_position = last_patch_position
        self.loss_function = loss_function
        self.testing = testing
        self.confusion_matrix = confusion_matrix

    def on_train_begin(self, logs=None):
        if self.testing != "not":
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["patch", "loss", "accuracy", "mean_iou", "accuracy_bg", "accuracy_class_1", "accuracy_class_2"])

    def intersection_union_per_patch(self, y_true, y_pred, num_classes):
        intersections = np.zeros(num_classes, dtype=np.int64)
        unions = np.zeros(num_classes, dtype=np.int64)

        for c in range(num_classes):
            y_true_c = (y_true == c)
            y_pred_c = (y_pred == c)

            intersections[c] = np.logical_and(y_true_c, y_pred_c).sum()
            unions[c] = np.logical_or(y_true_c, y_pred_c).sum()

        return intersections, unions

    def test_test_dataset(self):
        total_loss = 0.0
        total_correct = 0
        total_voxels = 0
        all_preds = []
        all_gt = []
        cm = None

        total_intersections = np.zeros(self.num_classes)
        total_unions = np.zeros(self.num_classes)
        current_patch_position = [0, 0, 0]

        for current_patch_number in range(self.total_patches):

            test_dataset, current_patch_position = self.test_patching.create_patch_dataset(
                current_patch_number,
                current_patch_position,
                self.last_patch_position
            )

            test_dataset = self.test_patching.shuffle(test_dataset, self.seed, 1)
            for image_patch, gt_patch in test_dataset.take(1):
                pred_logits = self.model(image_patch, training=False)
                loss_value = self.loss_function(gt_patch, pred_logits).numpy()

                pred = tf.argmax(pred_logits, axis=-1)
                pred_np = pred.numpy()
                gt_np = gt_patch.numpy()

                pred_np = np.squeeze(pred_np, axis=0)
                gt_np = np.squeeze(gt_np, axis=0)

                gt_np = np.squeeze(gt_np, axis=-1)

                if self.confusion_matrix is not None:
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

        mean_loss = total_loss / total_voxels
        mean_accuracy = total_correct / total_voxels

        iou_per_class = total_intersections / (total_unions + 1e-7)
        valid_classes = total_unions > 0
        mean_iou = np.mean(iou_per_class[valid_classes]) if np.any(valid_classes) else 0.0

        if self.confusion_matrix is not None:
            all_preds_flat = np.concatenate(all_preds).astype(np.int32)
            all_gt_flat = np.concatenate(all_gt).astype(np.int32)

            cm = confusion_matrix(all_gt_flat, all_preds_flat, labels=list(range(self.num_classes)))
            per_class_accuracy = cm.diagonal() / (cm.sum(axis=1) + 1e-7)
        return mean_loss, mean_accuracy, mean_iou, cm, per_class_accuracy

    def on_train_end(self, logs=None):
        if self.testing != "not":
            test_loss, test_accuracy, test_mean_iou, cm, per_class_acc = self.test_test_dataset()

            if cm is None:
                with open(self.csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.patch, test_loss, test_accuracy, test_mean_iou, -1, -1, -1])

                print(f"Test: Patch {self.patch}, Loss {test_loss}, Accuracy {test_accuracy}, Mean IoU {test_mean_iou}")

            else:
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")

                plt.savefig(self.cm_path)
                plt.close()

                with open(self.csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.patch, test_loss, test_accuracy, test_mean_iou, per_class_acc[0], per_class_acc[1], per_class_acc[2]])

                print(f"Test: Patch {self.patch}, Loss {test_loss}, Accuracy {test_accuracy}, Mean IoU {test_mean_iou}, Accuracy Background {per_class_acc[0]}, Accuracy Class 1 {per_class_acc[1]}, Accuracy Class 2 {per_class_acc[2]}")

    def set_model(self, model):
        super().set_model(model)



