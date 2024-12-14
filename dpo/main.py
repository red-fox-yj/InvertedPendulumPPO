import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =========================================
# 偏好数据集 (Preference Dataset)
# =========================================
class PreferenceDataset(Dataset):
    def __init__(self, data):
        """
        初始化偏好数据集。
        data: 每条数据格式为 (state, action_positive, action_negative)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# =========================================
# 偏好模型 (Preference Model)
# =========================================
class PreferenceModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        偏好模型：输入状态和动作对，输出两个动作的偏好分数。
        """
        super(PreferenceModel, self).__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        self.comparator = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出偏好分数
        )

    def forward(self, state, action1, action2):
        """
        对输入的状态和两组动作对，计算每个动作的偏好分数。
        """
        state_feat = self.state_encoder(state)
        action1_feat = self.action_encoder(action1)
        action2_feat = self.action_encoder(action2)
        
        # 拼接状态和动作特征，分别计算两组动作的偏好分数
        input1 = torch.cat([state_feat, action1_feat], dim=-1)
        input2 = torch.cat([state_feat, action2_feat], dim=-1)
        
        score1 = self.comparator(input1)
        score2 = self.comparator(input2)
        return score1, score2

# =========================================
# 策略模型 (Policy Model)
# =========================================
class PolicyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        策略模型：输入状态，输出动作分布（用于采样）。
        """
        super(PolicyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """
        输出每个动作的概率分布（Softmax）。
        """
        logits = self.fc(state)
        return torch.softmax(logits, dim=-1)

# =========================================
# 训练偏好模型
# =========================================
def train_preference_model(preference_model, dataset, epochs=10, batch_size=64, lr=1e-3):
    """
    使用偏好数据训练偏好模型。
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(preference_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for state, action1, action2, label in dataloader:
            # label: 偏好标签 (0: action2 优，1: action1 优)
            state = state.float()
            action1 = action1.float()
            action2 = action2.float()
            label = label.long()

            # 前向传播
            score1, score2 = preference_model(state, action1, action2)
            scores = torch.cat([score2, score1], dim=1)  # 拼成 [batch_size, 2]

            # 计算损失
            loss = criterion(scores, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# =========================================
# 更新策略模型
# =========================================
def train_policy_model(policy_model, preference_model, state_dim, action_dim, epochs=10, batch_size=64, lr=1e-3):
    """
    使用偏好模型训练策略模型，使策略输出符合偏好模型的预期。
    """
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for _ in range(batch_size):
            # 随机生成状态
            state = torch.randn((1, state_dim))
            
            # 从当前策略中采样动作
            action_dist = policy_model(state)
            action = torch.multinomial(action_dist, 1).squeeze(0)

            # 对策略的动作进行偏好评估
            action_one_hot = torch.eye(action_dim)[action].unsqueeze(0)  # 转为one-hot格式
            score1, score2 = preference_model(state, action_one_hot, torch.zeros_like(action_one_hot))

            # 偏好分数越高越好
            preference_score = score1 - score2
            loss = -preference_score.mean()  # 负号表示最大化偏好

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# =========================================
# 主程序
# =========================================
if __name__ == "__main__":
    # 假设状态维度为4，动作维度为3
    state_dim = 4
    action_dim = 3

    # 生成假偏好数据 (state, action1, action2, label)
    # label=1 表示 action1 更优，label=0 表示 action2 更优
    num_samples = 1000
    data = []
    for _ in range(num_samples):
        state = np.random.randn(state_dim)
        action1 = np.random.randn(action_dim)
        action2 = np.random.randn(action_dim)
        label = np.random.choice([0, 1])
        data.append((state, action1, action2, label))
    
    dataset = PreferenceDataset(data)

    # 初始化偏好模型和策略模型
    preference_model = PreferenceModel(state_dim, action_dim)
    policy_model = PolicyModel(state_dim, action_dim)

    # 训练偏好模型
    print("Training preference model...")
    train_preference_model(preference_model, dataset, epochs=5, batch_size=32, lr=1e-3)

    # 更新策略模型
    print("Training policy model...")
    train_policy_model(policy_model, preference_model, state_dim, action_dim, epochs=5, batch_size=32, lr=1e-3)

    # 测试策略模型
    print("Testing policy model...")
    state = torch.randn((1, state_dim))
    action_dist = policy_model(state)
    print("Action probabilities:", action_dist.detach().numpy())