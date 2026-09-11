import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import networkx as nx
import os
import matplotlib.pyplot as plt

os.makedirs('visuals', exist_ok=True)

# Dark Nostalgic War Template Settings for Plotly
WAR_THEME_LAYOUT = dict(
    paper_bgcolor='#12100e',
    plot_bgcolor='#1c1815',
    font=dict(family='Courier New, monospace, Georgia', color='#d4c5b9'),
    title_font=dict(family='Georgia, serif', size=20, color='#e6c88b')
)

# 1. Timeline Visualization for Target Countries
df = pd.read_csv('data/cleaned_data.csv')
targets = ['Sudan', 'Libya', 'Rwanda', 'Zimbabwe', 'Germany', 'Italy', 'USSR']
df_targets = df[df['Entity'].isin(targets)].copy()

# Add hover text formatting
df_targets['hover_text'] = df_targets.apply(
    lambda r: f"<b>{r['Entity']} ({int(r['Year'])})</b><br>"
              f"Polity Score: {r['polity_score']}<br>"
              f"GDP per Capita: ${r['gdp_pc']:,.0f}<br>"
              f"Conflict Deaths: {int(r['conflict_deaths']):,}<br>"
              f"Econ Collapse: {'YES' if r['econ_collapse']==1 else 'NO'}",
    axis=1
)

fig_timeline = px.scatter(
    df_targets,
    x='Year',
    y='Entity',
    size='conflict_deaths',
    color='polity_score',
    hover_name='Entity',
    hover_data={'Year': True, 'polity_score': True, 'gdp_pc': ':.0f', 'conflict_deaths': ':,', 'Entity': False},
    title='<b>HISTORICAL TIMELINE: REGIME POLITY & CONFLICT DEATHS</b>',
    color_continuous_scale='RdYlBu_r',
    size_max=45
)

fig_timeline.update_layout(
    **WAR_THEME_LAYOUT,
    xaxis=dict(gridcolor='#332a24', title='Year'),
    yaxis=dict(gridcolor='#332a24', title='Nation / Entity'),
    coloraxis_colorbar=dict(title="Polity Score<br>(-10 Autocracy to +10 Democracy)")
)
fig_timeline.write_html('visuals/timeline.html')
print("Timeline written to visuals/timeline.html")

# 2. Interactive Map of Instability Risk (choropleth)
latest_df = df[df['Year'] == 2022].copy()
latest_df['risk_score'] = (10 - latest_df['polity_score'].fillna(0)) + (latest_df['conflict_deaths'] > 0).astype(int) * 5 + (latest_df['econ_collapse'] * 3)

fig_map = px.choropleth(
    latest_df,
    locations="Code",
    color="risk_score",
    hover_name="Entity",
    hover_data={'polity_score': True, 'conflict_deaths': True, 'gdp_pc': ':.0f'},
    color_continuous_scale=['#2b2b2b', '#8b5a2b', '#c0392b', '#e74c3c', '#ff2a00'],
    title='<b>GLOBAL INSTABILITY & WAR RISK INDICATOR (2022 MATRIX)</b>'
)

fig_map.update_geos(
    showcoastlines=True, coastlinecolor="#4a3b32",
    showland=True, landcolor="#1e1a17",
    showocean=True, oceancolor="#0b0a09",
    showlakes=True, lakecolor="#0b0a09",
    showcountries=True, countrycolor="#3a2f28"
)

fig_map.update_layout(
    **WAR_THEME_LAYOUT,
    margin=dict(l=0, r=0, t=50, b=0),
    coloraxis_colorbar=dict(title="Instability Risk Index")
)
fig_map.write_html('visuals/risk_map.html')
print("Risk map written to visuals/risk_map.html")

# 3. Interactive War Interactions Map (Vector lines & Conflict Arcs)
with open('data/war_interactions.json', 'r') as f:
    war_interactions = json.load(f)

fig_war = go.Figure()

# Add country points/markers
for inter in war_interactions:
    # Origin country marker
    fig_war.add_trace(go.Scattergeo(
        lon=[inter['from_coords'][1]],
        lat=[inter['from_coords'][0]],
        mode='markers+text',
        name=inter['from_country'],
        text=[f"<b>{inter['from_country']}</b>"],
        textposition="top center",
        marker=dict(size=12, color='#e74c3c', symbol='diamond', line=dict(color='#ffeaa7', width=1)),
        hoverinfo='text',
        hovertext=f"<b>Origin: {inter['from_country']}</b><br>Conflict: {inter['name']}<br>Era: {inter['era']}"
    ))

    # Target country marker
    fig_war.add_trace(go.Scattergeo(
        lon=[inter['to_coords'][1]],
        lat=[inter['to_coords'][0]],
        mode='markers+text',
        name=inter['to_country'],
        text=[f"<b>{inter['to_country']}</b>"],
        textposition="bottom center",
        marker=dict(size=12, color='#d63031', symbol='cross', line=dict(color='#ffeaa7', width=1)),
        hoverinfo='text',
        hovertext=f"<b>Theater: {inter['to_country']}</b><br>Conflict: {inter['name']}<br>Estimated Deaths: {inter['estimated_deaths']:,}"
    ))

    # Curved/Line vector connecting origin to target
    lons = [inter['from_coords'][1], inter['to_coords'][1]]
    lats = [inter['from_coords'][0], inter['to_coords'][0]]

    hover_details = (
        f"<b>{inter['name']}</b> ({inter['era']})<br>"
        f"<b>Combatants:</b> {inter['from_country']} &rarr; {inter['to_country']}<br>"
        f"<b>Leaders Involved:</b> {', '.join(inter['key_leaders'])}<br>"
        f"<b>Casualties:</b> ~{inter['estimated_deaths']:,}<br>"
        f"<b>Key Battles:</b> {', '.join(inter['key_battles'])}<br>"
        f"<b>Causal Trigger:</b> {inter['causal_trigger']}"
    )

    fig_war.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(width=3, color='#e67e22'),
        opacity=0.85,
        name=inter['name'],
        hoverinfo='text',
        hovertext=hover_details
    ))

fig_war.update_geos(
    projection_type="natural earth",
    showcoastlines=True, coastlinecolor="#5a4738",
    showland=True, landcolor="#1c1815",
    showocean=True, oceancolor="#0c0a09",
    showcountries=True, countrycolor="#3a2f28",
    showlakes=True, lakecolor="#0c0a09"
)

fig_war.update_layout(
    title='<b>THEATER OF WAR: HISTORICAL WAR INTERACTIONS & CONFLICT ARCS</b>',
    **WAR_THEME_LAYOUT,
    showlegend=False,
    margin=dict(l=0, r=0, t=50, b=0)
)
fig_war.write_html('visuals/war_interactions.html')
print("War interactions map written to visuals/war_interactions.html")

# 4. Causal Graph Image (Static NetworkX diagram)
with open('data/causal_graph.json') as f:
    graph_data = json.load(f)

from networkx.readwrite import json_graph
G = json_graph.node_link_graph(graph_data)

plt.figure(figsize=(12, 9), facecolor='#12100e')
ax = plt.gca()
ax.set_facecolor('#12100e')

pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight'] / 8 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, node_size=3800, node_color="#2c221e", edgecolors="#e6c88b", linewidths=2)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="#e6c88b", font_family="serif")
nx.draw_networkx_edges(G, pos, width=weights, edge_color='#e67e22', arrowsize=25, arrowstyle='->', connectionstyle='arc3,rad=0.1')

plt.title("Causal Transition Graph: Economic Collapse -> Dictatorship -> War", color="#e6c88b", fontsize=14, fontweight="bold", pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('visuals/causal_map.png', facecolor='#12100e')
plt.close()
print("Causal map image written to visuals/causal_map.png")
