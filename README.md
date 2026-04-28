# World War Chain Reaction Model

This project builds a causal graph model to study how economic, political, and social events chain together into large-scale conflicts. It focuses on the ripple effects of small triggers (like economic collapse) leading to systemic failure, with a specific focus on African conflicts and historical dictatorships.

## Features

- **Causal Graph (NetworkX)**: Maps transitions between Economic Stability, Collapse, Dictatorship, and War.
- **AI Early Warning System**: A Random Forest model trained on historical data to predict conflict risk with ~96% accuracy.
- **Graph Machine Learning (GNN)**: Uses a Graph Convolutional Network to predict instability zones.
- **Interactive Visualizations**: Plotly-driven maps and timelines of conflict and regime changes.
- **Philosophical Reflection**: An analysis of "How small sparks lead to systemic collapse."

## Data Sources

- **Polity 5**: Political regime characteristics and transitions.
- **Maddison Project Database**: Historical GDP per capita.
- **Uppsala Conflict Data Program (UCDP)**: Deaths in armed conflicts.

## Project Structure

- `data/`: Raw and cleaned datasets, and saved model files.
- `visuals/`: Generated HTML and PNG visualizations.
- `clean_data.py`: Merges and cleans the raw datasets.
- `build_graph.py`: Constructs the causal transition graph.
- `visualize.py`: Generates the timeline, risk map, and causal map.
- `train_gnn.py`: Trains the Graph Neural Network.
- `early_warning.py`: Trains the predictive model and simulates future risk for Sudan, Libya, Rwanda, and Zimbabwe.
- `reflection.md`: Philosophical insights into the model's findings.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install networkx pandas matplotlib scikit-learn torch torch-geometric plotly joblib
   ```

2. **Execute the Pipeline**:
   ```bash
   python3 clean_data.py
   python3 build_graph.py
   python3 visualize.py
   python3 train_gnn.py
   python3 early_warning.py
   ```

3. **View Results**:
   - Check `visuals/` for interactive maps.
   - Read `reflection.md` for a deep dive into the causal mechanics of war.
