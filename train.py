from prepare import *
from model import *
from utils import *

b = 32
num_epochs = 60
train_folder = 'data/good'
lb = [4, 5, 6, 7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----fit function--------------------------------------------------------------
def fit(folder, lb, b, device, num_epochs):
    """
    folder <-- train folder 
    lb <-- lower bound
    b <-- batch size
    """
    #dataset and loader
    train_ds, test_ds, train_loader, test_loader = make_data_loaders(folder, lb, b)

    #initialise model
    model = TimeSeriesTransformer(
        input_dim=450,
        num_classes=2,
        d_model=64,  # 32 -> 64
        nhead=8,     # 4 -> 8
        num_layers=2,
        dim_feedforward=512,  # 256 -> 512
        dropout=0.2  # 0.4 -> 0.2 to allow more learning
    )
    model.to(device)

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.003,  # peak learning rate
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3  # percentage of training to increase LR
    )

    # tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    #train loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            # forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_ds)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(test_ds)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        # update learning rate
        scheduler.step(val_epoch_loss)
        
        print(f'epoch {epoch+1}/{num_epochs}: '
            f'train Loss: {epoch_loss:.4f}, train Acc: {epoch_acc:.4f}, '
            f'val Loss: {val_epoch_loss:.4f}, val Acc: {val_epoch_acc:.4f}')
        
    torch.save(model.state_dict(), f'results/transformer_lb_{lb}_epochs_{num_epochs}.pth')

    plot_loss(train_losses, val_losses, save_path=f'results/figs/loss_curves_lb_{lb}_epochs_{num_epochs}.png')
    plot_acc(train_accuracies, val_accuracies, save_path=f'results/figs/acc_curves_lb_{lb}_epochs_{num_epochs}.png')

    return model, test_loader

#----track accuracies-------------------------------------------------------
lb = [4, 5, 6, 7]
test_acc = {}

total_windows = 0 
total_time = 0

for _ in lb:
    model, test_loader = fit(train_folder, _, b, device, num_epochs)
    acc, w, t = model_eval(model, test_loader, device)
    test_acc[_] = acc
    total_time += t
    total_windows += w

acc_over_lb(test_acc, save_path=f'results/figs/acc_over_lb.png')
print(f'avg time taken per window: {total_windows/ total_time}')