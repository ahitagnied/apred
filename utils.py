import matplotlib.pyplot as plt
from model import *
from prepare import *
from sklearn.metrics import accuracy_score, classification_report
import time

#----train and test loader---------------------------------------------------
def make_data_loaders(folder, lb, b):
    entries = os.listdir(folder)
    train_filenames = [
        os.path.join(folder, fname)
        for fname in entries
        if os.path.isfile(os.path.join(folder, fname))
    ]

    prep = DataProcessing(train_filenames, lb)
    prep.load_files()
    prep.clean_data()
    prep.label_data()
    prep.build_features()
    X_train, X_test, y_train, y_test = prep.get_train_test_tensors()

    print("\n")

    print("X_train:", X_train.shape)  
    print("y_train:", y_train.shape)  
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    print("\n")

    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=b, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=b, shuffle=False)
    return train_ds, test_ds, train_loader, test_loader

#----plot training curves-------------------------------------------------------
def plot_loss(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.rc('font', family='serif')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14)  
    plt.title('Loss Curves', fontsize=18)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_acc(train_accuracies, val_accuracies, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.rc('font', family='serif')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=14)
    plt.title('Accuracy Curves', fontsize=18)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

#----model eval--------------------------------------------------------------
def model_eval(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    t0 = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_acc = accuracy_score(y_true, y_pred)
    # print final metrics
    print("\nfinal test accuracy:", test_acc)
    print("\nclassification Report:")
    print(classification_report(y_true, y_pred))
    t1 = time.time()
    num_windows = len(y_true)
    total_time = t1-t0
    return test_acc, num_windows, total_time

#----test acc plot-------------------------------------------------------------
def acc_over_lb(data_dict, save_path=None):
    # extract labels and values in insertion order
    labels = list(data_dict.keys())
    values = [data_dict[k] for k in labels]
    plt.figure(figsize=(8, 5))
    plt.rc('font', family='serif')
    # plot line with circular markers
    plt.plot(labels, values, marker='o', linewidth=1.5)
    # axis labels and title
    plt.xlabel('lower bound', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.title('accuracy v/s latency', fontsize=18)
    # set y-limits
    plt.ylim(0.90, 1.00)
    # subtle dashed grid
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    # tick styling
    plt.tick_params(axis='both', which='major', labelsize=14, direction='out')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()