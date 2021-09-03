import torch
from titanic.CustomDataset import titanic_dataset_test
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)


# Evaluation
dataset_test_path = '../data/test.csv'
titanic_dataset_test = titanic_dataset_test(dataset_test_path)
test_inputs, test_id = titanic_dataset_test.x, titanic_dataset_test.id
test_inputs, test_id = test_inputs.to(device), test_id.to(device)
net = torch.load('./model/model87+')
net.eval()
net.to(device)
torch.no_grad()
test_outputs = net(test_inputs)
test_predicted = torch.heaviside(test_outputs - torch.tensor(0.5), torch.tensor(1.))

submission = np.zeros((418, 2))
submission[:, 0], submission[:, 1]= np.squeeze(test_id.to('cpu').detach().numpy()), np.squeeze(test_predicted.to('cpu').detach().numpy())
submission = submission.astype(int)
submission = pd.DataFrame({'PassengerId': submission[:, 0], 'Survived': submission[:, 1]})
submission.to_csv('./submissions/submission.csv', index=False)