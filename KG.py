import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



def load_data(trip_file, poi_file):
    trips = pd.read_csv(trip_file)
    pois = pd.read_csv(poi_file)
    trips['time_of_day'] = pd.to_datetime(trips['Trip Start Timestamp']).dt.hour
    trips['day_of_week'] = pd.to_datetime(trips['Trip Start Timestamp']).dt.dayofweek
    trips['time_bin'] = pd.to_datetime(trips['Trip Start Timestamp']).dt.floor('15T')
    return trips, pois


#Construct Knowledge Graph
def construct_kg(trips, pois, time_bins, fares):
    # Replace NaN values with default values
    trips['Pickup Community Area'].fillna(-1, inplace=True)
    trips['Dropoff Community Area'].fillna(-1, inplace=True)
    trips['Rides Pooled'].fillna(0, inplace=True)

    G = nx.MultiDiGraph()

    # Process trips
    for _, row in trips.iterrows():
        try:
            # Ensure community area IDs are integers
            pickup_area = int(float(row['Pickup Community Area']))
            dropoff_area = int(float(row['Dropoff Community Area']))
            trip_id = f"trip_{row['Trip ID']}"
            G.add_node(trip_id, type='trip')

            if pickup_area != -1:  # Add pickup area relation
                pickup_node = f"area_{pickup_area}"
                G.add_node(pickup_node, type='area', name=row.get('Pickup Area Name', 'Unknown'))
                G.add_edge(trip_id, pickup_node, relation='origin_at')

            if dropoff_area != -1:  # Add dropoff area relation
                dropoff_node = f"area_{dropoff_area}"
                G.add_node(dropoff_node, type='area', name=row.get('Dropoff Area Name', 'Unknown'))
                G.add_edge(trip_id, dropoff_node, relation='destination')

            # Add rides pooled relationship
            rides_pooled = int(row['Rides Pooled'])
            G.add_node(f"rides_{rides_pooled}", type='rides_pooled')
            G.add_edge(trip_id, f"rides_{rides_pooled}", relation='rides_pooled')

        except ValueError as e:
            print(f"Error processing trip row: {row}. Error: {e}")

    # Process POIs
    for _, row in pois.iterrows():
        try:
            community_area = int(float(row['AREA_NUMBE']))
            poi_node = f"poi_{row['name']}"
            G.add_node(poi_node, type='poi', category=row['cate'])
            G.add_edge(poi_node, f"area_{community_area}", relation='located_at')

        except ValueError as e:
            print(f"Error processing POI row: {row}. Error: {e}")

    # Add time bins as nodes and relationships
    for _, row in time_bins.iterrows():
        trip_id = f"trip_{row['Trip ID']}"
        time_bin_node = f"time_bin_{row['Time Bin']}"
        G.add_node(time_bin_node, type='time_bin')
        G.add_edge(trip_id, time_bin_node, relation='starts_at')

    # Add fare ranges as nodes and relationships
    for _, row in fares.iterrows():
        trip_id = f"trip_{row['Trip ID']}"
        fare_node = f"fare_{row['Fare Range']}"
        G.add_node(fare_node, type='fare')
        G.add_edge(trip_id, fare_node, relation='fared_at')

    # Add adjacency between areas
    for area1, area2 in trips[['Pickup Community Area', 'Dropoff Community Area']].drop_duplicates().values:
        if area1 != -1 and area2 != -1 and area1 != area2:
            G.add_edge(f"area_{int(area1)}", f"area_{int(area2)}", relation='adjacent')

    return G


# Graph Embedding
class GCNEmbedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEmbedding, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Downstream task: Demand Prediction Model
class DemandPredictionModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DemandPredictionModel, self).__init__()
        self.gcn = GCNEmbedding(in_channels, hidden_channels, hidden_channels)
        self.fc_time = torch.nn.Linear(2, hidden_channels)  # For time features (time of day, day of week)
        self.fc = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index, time_features):
        x_gcn = self.gcn(x, edge_index)
        x_time = F.relu(self.fc_time(time_features))
        x_combined = torch.cat([x_gcn, x_time], dim=1)
        x_out = self.fc(x_combined)
        return x_out



#Prepare Demand Data
def prepare_demand_data(trips):

    trips['time_of_day'] = pd.to_datetime(trips['time_bin']).dt.hour
    trips['day_of_week'] = pd.to_datetime(trips['time_bin']).dt.dayofweek

    # Group by community area and 15-minute intervals
    demand = trips.groupby(['Pickup Community Area', 'time_bin']).size().reset_index(name='demand')

    # Add temporal features to the demand DataFrame
    temporal_features = trips[['time_bin', 'time_of_day', 'day_of_week']].drop_duplicates()

    demand_pivot = demand.pivot(index='time_bin', columns='Pickup Community Area', values='demand').fillna(0)

    demand_pivot = demand_pivot.merge(temporal_features.set_index('time_bin'), left_index=True, right_index=True)
    demand_columns = demand_pivot.columns.difference(['time_of_day', 'day_of_week'])
    demand_pivot[demand_columns] = (demand_pivot[demand_columns] - demand_pivot[demand_columns].min()) / (
        demand_pivot[demand_columns].max() - demand_pivot[demand_columns].min()
    )

    return demand_pivot


#Convert Knowledge Graph to PyTorch Geometric Format
def convert_to_pytorch_geo(kg, demand):
    node_mapping = {node: idx for idx, node in enumerate(kg.nodes)}
    edge_index = []
    for source, target, _ in kg.edges(data=True):
        edge_index.append([node_mapping[source], node_mapping[target]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = []
    demand_nodes = []
    aligned_columns = []

    for node, data in kg.nodes(data=True):
        feature = [
            1 if data['type'] == 'trip' else 0,
            1 if data['type'] == 'area' else 0,
            1 if data['type'] == 'poi' else 0,
        ]
        x.append(feature)

        if data['type'] == 'area':
            try:
                area_id = int(node.split("_")[1])
                if area_id in demand.columns:
                    demand_nodes.append(node_mapping[node])
                    aligned_columns.append(area_id)
            except ValueError:
                print(f"Invalid area ID for node: {node}")

    x = torch.tensor(x, dtype=torch.float)

    demand = demand[aligned_columns + ['time_of_day', 'day_of_week']]
    y = torch.tensor(demand[aligned_columns].values, dtype=torch.float)

    time_features = []
    for node, data in kg.nodes(data=True):
        if data['type'] == 'area':
            time_features.append(demand[['time_of_day', 'day_of_week']].values.mean(axis=0))  
        else:
            time_features.append([0, 0])  
    time_features = torch.tensor(time_features, dtype=torch.float)

    print(f"Aligned Columns: {aligned_columns}")
    print(f"Demand Nodes: {demand_nodes}")
    print(f"Demand Labels Shape: {y.shape}")
    print(f"Time Features Shape: {time_features.shape}")

    return x, edge_index, y, demand_nodes, time_features


#Training and Evaluate
def train_model(model, data, demand_nodes, epochs, optimizer, criterion, time_features):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        losses = []
        for t in range(data.y.shape[0]):  
            timestep_features = time_features[t].repeat(data.x.size(0), 1)  # Expand to match all nodes
            
            out = model(data.x, data.edge_index, timestep_features).squeeze()
            relevant_out = out[demand_nodes]
            loss = criterion(relevant_out, data.y[t])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")


def evaluate_model(model, data, demand_nodes, time_features):
    model.eval()
    with torch.no_grad():
        y_true_list = []
        y_pred_list = []
        
        for t in range(data.y.shape[0]):  # Loop through timesteps
            # Repeat temporal features for all graph nodes
            timestep_features = time_features[t].repeat(data.x.size(0), 1)  # Expand to match all nodes
            
            out = model(data.x, data.edge_index, timestep_features).squeeze()
            relevant_out = out[demand_nodes]
            y_pred_list.append(relevant_out.numpy())
            y_true_list.append(data.y[t].numpy())
        
        y_true = np.vstack(y_true_list)
        y_pred = np.vstack(y_pred_list)
        
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = mean_squared_error(y_true.flatten(), y_pred.flatten(), squared=False)
        
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return mae, rmse


# def visualize_kg(kg, save_path="knowledge_graph.png"):
#     pos = nx.spring_layout(kg)
#     plt.figure(figsize=(12, 12))
#     nx.draw(kg, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8)
#     plt.title("Knowledge Graph Visualization")
#     plt.savefig(save_path)
#     plt.show()


# Main Execution
if __name__ == "__main__":
    
    trip_data, poi_data = load_data('chicago_trip_data.csv', 'POIs.csv')

    kg = construct_kg(trip_data, poi_data)
    print(f"Knowledge Graph Statistics:")
    print(f"  Total Nodes: {kg.number_of_nodes()}")
    print(f"  Total Edges: {kg.number_of_edges()}")


    # Prepare demand data
    demand = prepare_demand_data(trip_data)
    time_features = torch.tensor(demand[['time_of_day', 'day_of_week']].values, dtype=torch.float)

    x, edge_index, y, demand_nodes, time_features = convert_to_pytorch_geo(kg, demand)
    
    print(f"Number of Demand Nodes: {len(demand_nodes)}")
    print(f"Number of Labels (y): {y.shape[1]}")

    # Prepare data for training
    data_with_kg = type('Data', (object,), {'x': x, 'edge_index': edge_index, 'y': y})()
    data_without_kg = type('Data', (object,), {'x': x[:, :10], 'edge_index': edge_index, 'y': y})()  # Remove KG embeddings

    # Initialize model, optimizer, and criterion
    model_with_kg = DemandPredictionModel(in_channels=data_with_kg.x.size(1), hidden_channels=64, out_channels=1)
    model_without_kg = DemandPredictionModel(in_channels=data_without_kg.x.size(1), hidden_channels=64, out_channels=1)

    optimizer_with_kg = torch.optim.Adam(model_with_kg.parameters(), lr=0.01)
    optimizer_without_kg = torch.optim.Adam(model_without_kg.parameters(), lr=0.01)

    criterion = torch.nn.MSELoss()
    


    # Train and evaluate models
    train_model(model_with_kg, data_with_kg, demand_nodes, epochs=20, optimizer=optimizer_with_kg, criterion=criterion, time_features=time_features)
    train_model(model_without_kg, data_without_kg, demand_nodes, epochs=20, optimizer=optimizer_without_kg, criterion=criterion, time_features=time_features)


    mae_with_kg, rmse_with_kg = evaluate_model(model_with_kg, data_with_kg, demand_nodes, time_features)
    mae_without_kg, rmse_without_kg = evaluate_model(model_without_kg, data_without_kg, demand_nodes, time_features)

    print(f"With KG Embeddings - MAE: {mae_with_kg}, RMSE: {rmse_with_kg}")
    print(f"Without KG Embeddings - MAE: {mae_without_kg}, RMSE: {rmse_without_kg}")
