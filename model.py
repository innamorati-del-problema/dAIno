import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Linear(input_size, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = self.linear_stack(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.linear_stack, file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.RMSprop(model.parameters())
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + (self.gamma *
                                       torch.max(self.target_model(next_state[idx])))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
