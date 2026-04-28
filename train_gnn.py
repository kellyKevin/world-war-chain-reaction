import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 1. Load data
df = pd.read_csv('data/cleaned_data.csv')
latest_year = 2018 # Using 2018 as we have good coverage there
year_df = df[df['Year'] == latest_year].dropna(subset=['polity_score', 'gdp_pc', 'Code'])

# 2. Create country-to-index mapping
countries = year_df['Code'].unique()
country_to_idx = {code: i for i, code in enumerate(countries)}
idx_to_country = {i: code for code, i in country_to_idx.items()}

# 3. Define Edges (Simplification: Neighbors based on similar GDP or Region)
# Ideally we'd use borders, but let's use a "similarity" graph or mock borders for GNN demo
# Let's try to find a simple border dataset or use regional grouping.
# For simplicity in this environment, we'll connect countries in the same 'Entity' if they were regions,
# but here let's just use a random sparse graph or a k-NN on GDP if we can't find borders.
# Actually, let's just make it a chain for temporal or a cluster for regions.
# Better: Connect countries if they are in the same region (using OWID data usually has regions)

edge_index = []
# Dummy edges: connect index i to i+1 for demonstration if no better data
for i in range(len(countries) - 1):
    edge_index.append([i, i+1])
    edge_index.append([i+1, i])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 4. Features: [polity_score, log_gdp, econ_collapse]
x = []
for code in countries:
    row = year_df[year_df['Code'] == code].iloc[0]
    x.append([row['polity_score'], np.log(row['gdp_pc']), row['econ_collapse']])
x = torch.tensor(x, dtype=torch.float)

# 5. Target: Predict if war in next 5 years (using historical data to label)
# Let's say we want to predict if 'is_war' will be 1 in 2022
target_year = 2022
target_df = df[df['Year'] == target_year]
y = []
for code in countries:
    t_row = target_df[target_df['Code'] == code]
    if not t_row.empty:
        y.append(t_row.iloc[0]['is_war'])
    else:
        y.append(0)
y = torch.tensor(y, dtype=torch.long)

# 6. Define GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 7. Train
data = Data(x=x, edge_index=edge_index, y=y)
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 8. Save
torch.save(model.state_dict(), 'data/gnn_model.pt')
print("GNN training complete. Model saved to data/gnn_model.pt")
