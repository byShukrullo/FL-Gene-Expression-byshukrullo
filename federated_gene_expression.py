import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import random
import time 
 
# 1. GENE EXPRESSION DATA SIMULATOR
class GeneExpressionSimulator:
    """Genlarning ifodasi ma'lumotlarini simulyatsiya qilish uchun"""
    
    def __init__(self, n_genes=1000, n_samples=500):
        self.n_genes = n_genes
        self.n_samples = n_samples
    
    def generate_data(self, client_id: int, disease_bias: float = 0.1):
        """Har bir client uchun unique gene expression data yaratadi"""
        # Har xil gen ekspressiya patternlari
        np.random.seed(42 + client_id)
        
        # Asosiy gen ekspressiya ma'lumotlari
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_genes,
            n_informative=100,
            n_redundant=50,
            n_classes=2,  # Healthy vs Disease
            class_sep=0.8 + disease_bias,
            random_state=42 + client_id
        )
        
        # Gene expression qiymatlarini realistic qilish
        X = np.abs(X) * np.random.exponential(2, X.shape)
        
        # Gene nomlari
        gene_names = [f"GENE_{i+1:04d}" for i in range(self.n_genes)]
        
        return X, y, gene_names

# 2. STREAMING DATA GENERATOR
class StreamingGeneData:
    """Real-time gene expression data stream simulyatsiyasi"""
    
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.current_index = 0
        
    def get_next_batch(self):
        """Keyingi batch ma'lumotlarni qaytaradi"""
        if self.current_index >= len(self.X):
            self.current_index = 0  # Cycle through data
            
        end_index = min(self.current_index + self.batch_size, len(self.X))
        batch_X = self.X[self.current_index:end_index]
        batch_y = self.y[self.current_index:end_index]
        
        self.current_index = end_index
        return batch_X, batch_y

# 3. NEURAL NETWORK MODEL
class GeneExpressionNN(nn.Module):
    """Gene expression classification uchun neural network"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=2):
        super(GeneExpressionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# 4. FEDERATED LEARNING CLIENT
class FederatedClient:
    """Har bir hospital/lab uchun federated learning client"""
    
    def __init__(self, client_id: int, data_simulator: GeneExpressionSimulator):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ma'lumotlarni yaratish
        X, y, self.gene_names = data_simulator.generate_data(client_id, 
                                                           disease_bias=random.uniform(-0.2, 0.2))
        
        # Ma'lumotlarni preprocessing
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Streaming data yaratish
        self.streaming_data = StreamingGeneData(X_train, y_train)
        self.test_data = (torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        # Model va optimizer
        self.model = GeneExpressionNN(input_size=X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Client {client_id} initialized with {len(X)} samples")
    
    def local_train(self, global_weights: Dict = None, epochs: int = 5):
        """Local training with streaming data"""
        if global_weights:
            self.model.load_state_dict(global_weights)
        
        self.model.train()
        total_loss = 0
        batches_processed = 0
        
        for epoch in range(epochs):
            # Bir nechta batch lar bilan train qilish
            for _ in range(10):  # Har epoch da 10 ta batch
                batch_X, batch_y = self.streaming_data.get_next_batch()
                
                if len(batch_X) == 0:
                    continue
                
                # Tensor ga o'tkazish
                batch_X = torch.FloatTensor(batch_X).to(self.device)
                batch_y = torch.LongTensor(batch_y).to(self.device)
                
                # Training step
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches_processed += 1
        
        avg_loss = total_loss / max(batches_processed, 1)
        return self.model.state_dict(), avg_loss
    
    def evaluate(self):
        """Test ma'lumotlari ustida model baholash"""
        self.model.eval()
        X_test, y_test = self.test_data
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
        
        return accuracy

# 5. FEDERATED LEARNING SERVER
class FederatedServer:
    """Central server for federated learning coordination"""
    
    def __init__(self, model_template: nn.Module):
        self.global_model = model_template
        self.client_weights = []
        self.global_weights = self.global_model.state_dict()
        self.training_history = {
            'rounds': [],
            'avg_accuracy': [],
            'avg_loss': []
        }
    
    def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]):
        """FedAvg algorithm - client weights larni aggregate qilish"""
        total_size = sum(client_sizes)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Har bir parameter uchun weighted average hisoblash
        for key in client_weights[0].keys():
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key])
            
            for i, weights in enumerate(client_weights):
                weight = client_sizes[i] / total_size
                aggregated_weights[key] = aggregated_weights[key].float()
                aggregated_weights[key] += weights[key] * weight

        
        self.global_weights = aggregated_weights
        return aggregated_weights
    
    def federated_round(self, clients: List[FederatedClient], round_num: int):
        """Bir round federated learning"""
        print(f"\n=== Federated Round {round_num} ===")
        
        client_weights = []
        client_losses = []
        client_accuracies = []
        client_sizes = []
        
        # Har bir client bilan local training
        for client in clients:
            print(f"Training Client {client.client_id}...")
            
            # Local training
            weights, loss = client.local_train(self.global_weights, epochs=3)
            accuracy = client.evaluate()
            
            client_weights.append(weights)
            client_losses.append(loss)
            client_accuracies.append(accuracy)
            client_sizes.append(len(client.streaming_data.X))
            
            print(f"  Client {client.client_id}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Weights larni aggregate qilish
        self.aggregate_weights(client_weights, client_sizes)
        
        # Global statistics
        avg_loss = np.mean(client_losses)
        avg_accuracy = np.mean(client_accuracies)
        
        # History ni saqlash
        self.training_history['rounds'].append(round_num)
        self.training_history['avg_loss'].append(avg_loss)
        self.training_history['avg_accuracy'].append(avg_accuracy)
        
        print(f"Round {round_num}: Avg Loss={avg_loss:.4f}, Avg Accuracy={avg_accuracy:.4f}")
        
        return avg_loss, avg_accuracy

# 6. VISUALIZATION VA MONITORING
class FederatedMonitor:
    """Federated learning jarayonini monitoring qilish"""
    
    @staticmethod
    def plot_training_history(history: Dict):
        """Training history ni vizualizatsiya qilish"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['rounds'], history['avg_loss'], 'b-', marker='o')
        ax1.set_title('Federated Learning - Average Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['rounds'], history['avg_accuracy'], 'g-', marker='o')
        ax2.set_title('Federated Learning - Average Accuracy')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_client_performance(clients: List[FederatedClient]):
        """Client larning performansini solishtirish"""
        client_ids = []
        accuracies = []
        
        for client in clients:
            accuracy = client.evaluate()
            client_ids.append(f"Client {client.client_id}")
            accuracies.append(accuracy)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(client_ids, accuracies)
        plt.title('Client Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Bar larning ustiga qiymatlarni yozish
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# 7. MAIN EXECUTION
def main():
    """Asosiy federated learning simulation"""
    print("🧬 Federated Learning for Streaming Gene Expression Analysis")
    print("=" * 60)
    
    # 1. Data simulator yaratish
    data_simulator = GeneExpressionSimulator(n_genes=500, n_samples=300)
    
    # 2. Bir nechta client yaratish (turli hospital/lab lar)
    num_clients = 4
    clients = []
    
    print(f"\n📊 Creating {num_clients} federated clients (hospitals/labs)...")
    for i in range(num_clients):
        client = FederatedClient(i+1, data_simulator)
        clients.append(client)
    
    # 3. Server yaratish
    template_model = GeneExpressionNN(input_size=500)
    server = FederatedServer(template_model)
    
    # 4. Federated learning jarayoni
    print(f"\n Starting Federated Learning with {num_clients} clients...")
    num_rounds = 10
    
    for round_num in range(1, num_rounds + 1):
        avg_loss, avg_accuracy = server.federated_round(clients, round_num)
        time.sleep(1)  # Simulate real-time delay
    
    # 5. Natijalarni visualizatsiya qilish
    print(f"\n Training completed! Visualizing results...")
    
    monitor = FederatedMonitor()
    monitor.plot_training_history(server.training_history)
    monitor.compare_client_performance(clients)
    
    # 6. Final results
    final_accuracy = server.training_history['avg_accuracy'][-1]
    print(f"\n Final Results:")
    print(f"   - Final Average Accuracy: {final_accuracy:.4f}")
    print(f"   - Total Rounds: {num_rounds}")
    print(f"   - Number of Clients: {num_clients}")
    
    # 7. Gene analysis namunasi
    print(f"\n Sample Gene Analysis:")
    sample_client = clients[0]
    print(f"   - Client 1 has {len(sample_client.gene_names)} genes")
    print(f"   - Top 5 genes: {sample_client.gene_names[:5]}")
    
    return server, clients

if __name__ == "__main__":
    # Federated learning ni ishga tushirish
    server, clients = main()
    
    print("\n Federated Learning simulation completed successfully!")
    print(" Gene expression data processed across multiple institutions")
    print(" Privacy preserved - no raw data was shared between clients")