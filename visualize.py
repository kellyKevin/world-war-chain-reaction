import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import networkx as nx

# 1. Timeline Visualization for Target Countries
df = pd.read_csv('data/cleaned_data.csv')
targets = ['Sudan', 'Libya', 'Rwanda', 'Zimbabwe', 'Germany']
df_targets = df[df['Entity'].isin(targets)]

# Create a heatmap of instability
fig_timeline = px.scatter(df_targets, x='Year', y='Entity',
                 size='conflict_deaths', color='polity_score',
                 hover_data=['gdp_pc', 'econ_collapse'],
                 title='Conflict Deaths and Polity Score Over Time',
                 color_continuous_scale='RdBu_r')
fig_timeline.write_html('visuals/timeline.html')

# 2. Causal Graph Visualization (Simple Static NetworkX for now, but saved as image)
import matplotlib.pyplot as plt

with open('data/causal_graph.json') as f:
    data = json.load(f)

from networkx.readwrite import json_graph
G = json_graph.node_link_graph(data)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
weights = [G[u][v]['weight'] / 10 for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
        font_size=10, font_weight="bold", width=weights, edge_color='gray',
        arrowsize=20)
plt.title("Causal Graph of Instability Triggers")
plt.savefig('visuals/causal_map.png')

# 3. Interactive Map of Current Instability Risk (based on latest data)
latest_df = df[df['Year'] == 2022].copy()
# Simple heuristic risk: low polity + high deaths
latest_df['risk_score'] = (10 - latest_df['polity_score']) + (latest_df['conflict_deaths'] > 0).astype(int) * 5

fig_map = px.choropleth(latest_df, locations="Code",
                    color="risk_score",
                    hover_name="Entity",
                    color_continuous_scale=px.colors.sequential.YlOrRd,
                    title='2022 Instability Risk Indicator')
fig_map.write_html('visuals/risk_map.html')

print("Visualizations generated in visuals/ directory.")
