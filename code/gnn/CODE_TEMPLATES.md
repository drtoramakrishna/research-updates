# GNN Code Templates & Snippets

Quick copy-paste templates for common GNN tasks.

## Template 1: Basic GNN Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load data
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

# Normalize features
data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

# Define model
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GNN(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        val_acc = (out[data.val_mask].argmax(1) == data.y[data.val_mask]).float().mean()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")

# Test
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    test_acc = (out[data.test_mask].argmax(1) == data.y[data.test_mask]).float().mean()
    print(f"Test Accuracy: {test_acc:.4f}")
```

## Template 2: Link Prediction

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class LinkPrediction(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index, edge_pair):
        # Get embeddings
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # Link prediction
        x_i = x[edge_pair[0]]
        x_j = x[edge_pair[1]]
        x_ij = torch.cat([x_i, x_j], dim=1)
        
        return self.mlp(x_ij).sigmoid()

# Usage
model = LinkPrediction(dataset.num_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# For each batch
edge_pairs = torch.randint(0, data.num_nodes, (2, 1000))
preds = model(data.x, data.edge_index, edge_pairs)
loss = F.binary_cross_entropy(preds.squeeze(), labels)
```

## Template 3: Graph Classification

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader

class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # GNN
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # Pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.lin(x)
        return x

# Load graph dataset
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='./data', name='PROTEINS')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
model = GraphClassifier(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters())

for batch in loader:
    model.train()
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index, batch.batch)
    loss = F.cross_entropy(out, batch.y)
    loss.backward()
    optimizer.step()
```

## Template 4: Custom GNN Layer

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class CustomGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # Mean aggregation
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Normalize by degree
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x = self.lin(x)
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out

# Usage
conv = CustomGNNConv(in_channels=64, out_channels=32)
x = conv(x, edge_index)
```

## Template 5: Multi-Task Learning

```python
class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_regression):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Task-specific heads
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.regressor = nn.Linear(hidden_channels, num_regression)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        
        # Multi-task outputs
        class_out = self.classifier(x)
        reg_out = self.regressor(x)
        
        return class_out, reg_out

# Training
for epoch in range(100):
    class_out, reg_out = model(data.x, data.edge_index)
    
    # Combined loss
    class_loss = F.cross_entropy(class_out[train_mask], class_labels)
    reg_loss = F.mse_loss(reg_out[train_mask], reg_targets)
    
    loss = class_loss + 0.5 * reg_loss
    loss.backward()
```

## Template 6: Attention Weight Visualization (GAT)

```python
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

class GATWithAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=8, concat=True)
        self.lin = nn.Linear(hidden_channels * 8, num_classes)
    
    def forward(self, x, edge_index):
        x, attention = self.gat(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x = self.lin(x)
        return x, attention

# Visualize attention
model.eval()
out, (edge_index, attention) = model(data.x, data.edge_index)

# Get attention for first node
node_id = 0
neighbors = edge_index[1][edge_index[0] == node_id]
attentions = attention[edge_index[0] == node_id]

plt.bar(neighbors.cpu().numpy(), attentions.mean(dim=1).cpu().detach().numpy())
plt.xlabel('Neighbor ID')
plt.ylabel('Attention Weight')
plt.show()
```

## Template 7: Model Checkpointing

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'accuracy': accuracy
}
torch.save(checkpoint, 'model_checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# Continue training
for epoch in range(epoch, total_epochs):
    # ... training code ...
```

## Template 8: Hyperparameter Search

```python
import itertools

# Grid search
hyperparams = {
    'hidden_channels': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.0, 0.5, 0.7],
    'lr': [0.001, 0.01, 0.1]
}

best_acc = 0
best_params = None

for params in itertools.product(*hyperparams.values()):
    param_dict = dict(zip(hyperparams.keys(), params))
    
    # Train with these parameters
    model = GNN(**param_dict)
    # ... training loop ...
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_params = param_dict

print(f"Best params: {best_params}")
print(f"Best accuracy: {best_acc}")

# Alternative: Use Optuna for Bayesian optimization
import optuna

def objective(trial):
    hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    
    # Train model
    model = GNN(hidden_channels=hidden_channels, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # ... training loop ...
    
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## Template 9: Data Augmentation

```python
from torch_geometric.transforms import Compose, AddSelfLoops, NormalizeFeatures
from torch_geometric.utils import dropout_edge

# Transform pipeline
transform = Compose([
    AddSelfLoops(),
    NormalizeFeatures()
])

data = transform(data)

# Edge dropout (augmentation)
def augment_data(data, p=0.1):
    edge_index, _ = dropout_edge(data.edge_index, p=p)
    return Data(x=data.x, edge_index=edge_index, y=data.y)

# Train with augmentation
for epoch in range(100):
    # Augment data each epoch
    augmented_data = augment_data(data)
    
    out = model(augmented_data.x, augmented_data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
```

## Template 10: Export Model

```python
# Export to ONNX for deployment
dummy_input = torch.randn(100, dataset.num_features)
dummy_edge_index = torch.randint(0, 100, (2, 200))

torch.onnx.export(
    model,
    (dummy_input, dummy_edge_index),
    "model.onnx",
    input_names=['node_features', 'edge_index'],
    output_names=['output'],
    dynamic_axes={
        'node_features': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'}
    }
)

# Load and use
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
```

---

All templates are ready to use! Modify parameters as needed for your specific problem.
