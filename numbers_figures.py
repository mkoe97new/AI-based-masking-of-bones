import matplotlib.pyplot as plt
import csv
import os

def plot_final_log_values(patch, train_callbacks, eval_callbacks, summary_log_dir, model_name):
    full_train_epoch = []
    full_train_accuracy = []
    full_train_loss = []
    first_run = True
    for train_callback_iter in train_callbacks:
        if first_run:
            full_train_epoch = train_callback_iter.get_train_epoch()
            full_train_accuracy = train_callback_iter.get_train_accuracy()
            full_train_loss = train_callback_iter.get_train_loss()
            first_run = False
        else:
            full_train_accuracy = [x + y for x, y in
                                   zip(full_train_accuracy, train_callback_iter.get_train_accuracy())]
            full_train_loss = [x + y for x, y in zip(full_train_loss, train_callback_iter.get_train_loss())]

    full_eval_epoch = []
    full_eval_accuracy = []
    full_eval_loss = []
    first_run = True
    for eval_callback_iter in eval_callbacks:
        if first_run:
            full_eval_epoch = eval_callback_iter.get_eval_epoch()
            full_eval_accuracy = eval_callback_iter.get_eval_accuracy()
            full_eval_loss = eval_callback_iter.get_eval_loss()
            first_run = False
        else:
            full_eval_accuracy = [x + y for x, y in zip(full_eval_accuracy, eval_callback_iter.get_eval_accuracy())]
            full_eval_loss = [x + y for x, y in zip(full_eval_loss, eval_callback_iter.get_eval_loss())]

    full_train_accuracy = [x / len(train_callbacks) for x in full_train_accuracy]
    full_train_loss = [x / len(train_callbacks) for x in full_train_loss]
    full_test_accuracy = [x / len(eval_callbacks) for x in full_eval_accuracy]
    full_test_loss = [x / len(eval_callbacks) for x in full_eval_accuracy]

    plot_graph(full_train_epoch, full_train_accuracy, "Full Training Accuracy ", "Epoch", "Accuracy",
                    "Full Train Acc", summary_log_dir, model_name)
    plot_graph(full_train_epoch, full_train_loss, "Full Training Loss ", "Epoch", "Loss", "Full Train Loss",
                    summary_log_dir, model_name)
    plot_graph(full_eval_epoch, full_test_accuracy, "Full Test Accuracy ", "Epoch", "Accuracy",
                    "Full Test Acc", summary_log_dir, model_name)
    plot_graph(full_eval_epoch, full_test_loss, "Full Test Loss ", "Epoch", "Loss", "Test Loss",
                    summary_log_dir, model_name)

    save_numbers(patch, full_train_epoch, full_train_accuracy, full_train_loss, full_eval_epoch, full_test_accuracy,
                 full_test_loss, summary_log_dir, model_name)


def plot_graph(x_array, y_array, title, x_label, y_label, name, summary_log_dir, model_name):
    callback_dir = summary_log_dir + "/" + model_name + "/"
    fig = plt.figure()
    plt.plot(x_array, y_array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(callback_dir + name + '.png')
    plt.close(fig)

def save_numbers(patch, full_train_epoch, full_train_accuracy, full_train_loss, full_eval_epoch, full_eval_accuracy,
                 full_eval_loss, summary_log_dir, model_name):
    train_patch = []
    eval_patch = []
    for i in full_train_epoch:
        train_patch.append(patch)
    for i in full_train_epoch:
        eval_patch.append(patch)

    save_train_path = summary_log_dir + "/" + model_name + "/" + "train.csv"
    if os.path.exists(save_train_path):
        with open(save_train_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Patch', 'TrainEpoch', 'TrainAccuracy', 'TrainLoss'])

            # Combine arrays row-wise using zip and write each row
            for row in zip(train_patch, full_train_epoch, full_train_accuracy, full_train_loss):
                writer.writerow(row)
    else:
        with open(save_train_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Patch', 'TrainEpoch', 'TrainAccuracy', 'TrainLoss'])

            # Combine arrays row-wise using zip and write each row
            for row in zip(train_patch, full_train_epoch, full_train_accuracy, full_train_loss):
                writer.writerow(row)

    save_eval_path = summary_log_dir + "/" + model_name + "/" + "eval.csv"
    if os.path.exists(save_eval_path):
        with open(save_eval_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Patch', 'EvalEpoch', 'EvalAccuracy', 'EvalLoss'])

            # Combine arrays row-wise using zip and write each row
            for row in zip(eval_patch, full_eval_epoch, full_eval_accuracy, full_eval_loss):
                writer.writerow(row)
    else:
        save_eval_path = summary_log_dir + "/" + model_name + "/" + "eval.csv"
        with open(save_eval_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Patch', 'EvalEpoch', 'EvalAccuracy', 'EvalLoss'])

            # Combine arrays row-wise using zip and write each row
            for row in zip(eval_patch, full_eval_epoch, full_eval_accuracy, full_eval_loss):
                writer.writerow(row)