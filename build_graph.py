import pandas as pd
import networkx as nx
import json

df = pd.read_csv('data/cleaned_data.csv')

# We want to find chains: Econ Collapse -> Dictatorship -> War
# Let's shift indicators to see transitions

df = df.sort_values(['Entity', 'Year'])

# Create lagged versions to detect transitions
df['prev_econ_collapse'] = df.groupby('Entity')['econ_collapse'].shift(1)
df['prev_dictatorship'] = df.groupby('Entity')['is_dictatorship'].shift(1)
df['prev_war'] = df.groupby('Entity')['is_war'].shift(1)

# Define transitions
# 1. Economic Collapse -> Dictatorship (within 5 years)
# 2. Dictatorship -> War (within 5 years)
# 3. Economic Collapse -> War

# Simplified transition detection:
# If currently a dictatorship and was not one last year, did an economic collapse happen recently?
df['dictatorship_start'] = ((df['is_dictatorship'] == 1) & (df['prev_dictatorship'] == 0)).astype(int)
df['war_start'] = ((df['is_war'] == 1) & (df['prev_war'] == 0)).astype(int)

# Create the graph
G = nx.DiGraph()

nodes = ["Economic Stability", "Economic Collapse", "Democracy", "Dictatorship", "Peace", "War"]
G.add_nodes_from(nodes)

# Count occurrences for transition probabilities
# (This is a simplified abstraction)

transitions = {
    ("Economic Stability", "Economic Collapse"): 0,
    ("Economic Collapse", "Dictatorship"): 0,
    ("Dictatorship", "War"): 0,
    ("Economic Collapse", "War"): 0,
    ("Democracy", "Dictatorship"): 0,
    ("Peace", "War"): 0
}

# Example logic: How many times did a dictatorship start after an econ collapse in the last 3 years?
for entity in df['Entity'].unique():
    subset = df[df['Entity'] == entity]
    for i in range(len(subset)):
        if subset.iloc[i]['dictatorship_start'] == 1:
            # Look back 3 years for econ collapse
            lookback = subset.iloc[max(0, i-3):i]
            if lookback['econ_collapse'].any():
                transitions[("Economic Collapse", "Dictatorship")] += 1
            else:
                transitions[("Democracy", "Dictatorship")] += 1

        if subset.iloc[i]['war_start'] == 1:
            lookback = subset.iloc[max(0, i-5):i]
            if lookback['is_dictatorship'].any():
                transitions[("Dictatorship", "War")] += 1
            elif lookback['econ_collapse'].any():
                transitions[("Economic Collapse", "War")] += 1
            else:
                transitions[("Peace", "War")] += 1

# Add edges with weights
for (u, v), weight in transitions.items():
    if weight > 0:
        G.add_edge(u, v, weight=weight)

print("Nodes:", G.nodes())
print("Edges with weights:", G.edges(data=True))

# Save graph data for visualization
from networkx.readwrite import json_graph
data = json_graph.node_link_data(G)
with open('data/causal_graph.json', 'w') as f:
    json.dump(data, f)

print("Causal graph saved to data/causal_graph.json")
