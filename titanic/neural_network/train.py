import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
import math
from titanic.neural_network.CustomDataset import titanic_train_dataset
from titanic.neural_network.model import Net_test3
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


dataset_train_path = '../data/train.csv'
titanic_dataset = titanic_train_dataset(dataset_train_path)

val_num = math.ceil(len(titanic_dataset) * 0.2)
train_num = len(titanic_dataset) - val_num
train_set, val_set = random_split(titanic_dataset, (train_num, val_num))

# titanic_dataset.x = titanic_dataset.x[:840, :]
# titanic_dataset.y = titanic_dataset.y[:840, :]
# train_num = 700
# val_num = 140
# train_set, val_set = random_split(titanic_dataset, (train_num, val_num))
# print(f"Now you have {train_num} training samples and {val_num} validation samples")

batch_size=35
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


net = Net_test3()
net.to(device)  # Move model to GPU

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)


# Training Loop
num_epochs = 1000
num_show = 1
train_loss_list = []
val_loss_list = []
val_accuracy_list = []
stop = 0
for epoch in range(num_epochs):  # loop over the dataset multiple times
    if stop:
        break
    running_loss = 0.0
    for iteration, data in enumerate(train_dataloader):
        if len(val_accuracy_list) > 3 and np.all(np.array([val_accuracy_list[-i-1].item() for i in range(4)]) >= 0.87):
            torch.save(net, './model/model88+')
            stop = 1
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Zero the parameter gradients
        optimizer.zero_grad()

        # 2. Forward + 3. Calculate Loss + 4. Backward + 5. Optimize
        outputs = net(inputs)
        # outputs = outputs.squeeze()
        # labels = labels.squeeze(dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if iteration % num_show == (num_show - 1):    # print every num_show mini-batches
            running_loss = running_loss / num_show
            print('[%d, %5d] Train Loss: %.3f' %
                  (epoch + 1, iteration + 1, running_loss))
            train_loss_list.append(running_loss)
            running_loss = 0.0
            # Validation
            corrects = 0
            val_loss_sum = 0
            net.eval()
            with torch.no_grad():
                for val_iteration, val_data in enumerate(val_dataloader):
                    val_inputs, val_labels = val_data
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = net(val_inputs)
                    val_loss_sum += criterion(val_outputs, val_labels).item()
                    # val_predicted = torch.argmax(val_outputs, 1)
                    val_predicted = torch.heaviside(val_outputs - torch.tensor(0.5), torch.tensor(1.))
                    corrects += torch.sum(val_predicted == val_labels)
                val_accuracy = corrects / float(val_num)
                val_accuracy_list.append(val_accuracy)
                val_loss = val_loss_sum / len(val_dataloader)
                val_loss_list.append(val_loss)
                print(f'           Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')
            net.train()


iter_num = np.arange(1, len(train_loss_list) + 1) * 200
fig, ax1 = plt.subplots()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.plot(iter_num, train_loss_list, color='tab:blue', label='Train Loss')
ax1.plot(iter_num, val_loss_list, color='tab:red', label='Validation Loss')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
ax2.plot(iter_num, val_accuracy_list, color='tab:green', label='Validation Accuracy')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Best Iteration
# best_model1_iteration = val_accuracy_list.index(max(val_accuracy_list))
# best_model2_iteration = val_loss_list.index(min(val_loss_list))

print('finished')