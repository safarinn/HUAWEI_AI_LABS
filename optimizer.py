import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------
# 1. DATA (Synthetic)
# ------------------------
torch.manual_seed(42)

X = torch.unsqueeze(torch.linspace(-1, 1, 300), dim=1)
y = X**2 + 0.2 * torch.rand(X.size())

# ------------------------
# 2. MODEL (MLP)
# ------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# 3. TRAIN FUNCTION
# ------------------------
def train_model(optimizer_name, lr=0.01, epochs=50):
    model = SimpleMLP()  # reset weights
    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history

# ------------------------
# 4. RUN EXPERIMENTS
# ------------------------
sgd_loss = train_model("SGD")
adam_loss = train_model("Adam")
rms_loss = train_model("RMSprop")

# ------------------------
# 5. PLOT
# ------------------------
plt.plot(sgd_loss, label="SGD")
plt.plot(adam_loss, label="Adam")
plt.plot(rms_loss, label="RMSprop")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.show()