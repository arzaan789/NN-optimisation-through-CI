import numpy as np
from tqdm import tqdm
from lib import carpriceprediction

# Initialize evaluators
evaluate_train = carpriceprediction.modified_make_evaluator('train')
evaluate_val = carpriceprediction.modified_make_evaluator('validation')
evaluate_test = carpriceprediction.modified_make_evaluator('test')


# ==============================================
# Inner PSO: Weight Optimization
# ==============================================
class WeightParticle:
    def __init__(self, n_layers, hidden_units):
        """
        n_layers: number of hidden layers
        hidden_units: list of units per layer (length = n_layers)
        """
        self.n_layers = n_layers
        self.hidden_units = hidden_units

        # Calculate total parameters
        self.n_weights = self.calculate_parameters()

        # Position format: [n_layers, h1, h2,..., hn, weights..., biases..., w, c1, c2]
        self.position = np.concatenate([
            [n_layers],  # Number of layers
            hidden_units,  # Units per layer
            np.random.uniform(-5, 5, self.n_weights),  # Weights and biases
            np.random.uniform(0.1, 2, 3)  # PSO parameters
        ])
        self.velocity = np.zeros(len(self.position))
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def calculate_parameters(self):
        total = 0
        # Input -> First Hidden
        total += 21 * self.hidden_units[0]
        # Hidden -> Hidden
        for i in range(self.n_layers - 1):
            total += self.hidden_units[i] * self.hidden_units[i + 1]

        # Last Hidden -> Output
        total += self.hidden_units[-1] * 1
        # Biases (all hidden + output)
        total += sum(self.hidden_units) + 1
        return total

    def get_pso_params(self):
        return self.position[-3], self.position[-2], self.position[-1]  # [w, c1, c2]

    def update_velocity(self, global_best):
        w, c1, c2 = self.get_pso_params()
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        # Ensure hidden units remain integer
        self.position[0] = int(np.clip(round(self.position[0]), 1, 12))
        # Clip PSO params
        self.position[-3:] = np.clip(self.position[-3:], 0.1, 2.0)

    def evaluate(self):
        # Only pass weights/biases (exclude n_hidden and PSO params)
        mse = evaluate_train(self.position[:-3])
        return mse


def run_weight_pso(n_layers,n_hidden, n_particles=50, max_iter=50):
    swarm = [WeightParticle(n_layers,n_hidden) for _ in range(n_particles)]
    global_best = None
    global_best_value = float('inf')

    for _ in range(max_iter):
        for p in swarm:
            score = p.evaluate()
            if score < global_best_value:
                global_best_value = score
                global_best = np.copy(p.position)
                p.best_value = score
                p.best_position = np.copy(p.position)
            p.update_velocity(global_best)
            p.update_position()
    return global_best, global_best_value


# ==============================================
# Outer PSO: Architecture Optimization
# ==============================================
class ArchitectureParticle:
    def __init__(self, n_layers):
        # Position: [n_layers, [hidden_units1,hidden_unitsn], w, c1, c2]
        self.n_layers = n_layers

        self.position = np.concatenate([
            [self.n_layers],  # Hidden layers
            np.random.uniform(1, 25, n_layers),  # Hidden units
            np.random.uniform(0.1, 2, 3)  # PSO params
        ])
        self.velocity = np.zeros(len(self.position))
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def get_hidden_units(self):
        return [int(round(x)) for x in self.position[1:-3]]

    def update_velocity(self, global_best):
        w, c1, c2 = self.position[-3:]
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        # Clip all values from position[1:-3] to [1, 25]
        self.position[1:-3] = np.clip(np.round(self.position[1:-3]), 1, 25)
        # Clip PSO params
        self.position[-3:] = np.clip(self.position[-3:], 0.1, 2.0)


def run_architecture_pso(n_layers,n_particles=10, max_iter=10):
    swarm = [ArchitectureParticle(n_layers) for _ in range(n_particles)]
    global_best = None
    global_best_value = float('inf')

    for _ in range(max_iter):
        for p in swarm:
            n_hidden = p.get_hidden_units()
            best_weights, train_mse = run_weight_pso(n_layers,n_hidden)

            if train_mse < global_best_value:
                global_best_value = train_mse
                global_best = np.copy(p.position)
                p.best_value = train_mse
                p.best_position = np.copy(p.position)
            p.update_velocity(global_best)
            p.update_position()
    return global_best, global_best_value

class LayerParticle:

    def __init__(self):
        # Position: [n_layers, w, c1, c2]
        self.position = np.concatenate([
            [np.random.uniform(1, 15)],  # Number of layers
            np.random.uniform(0.1, 2, 3)  # PSO params
        ])
        self.velocity = np.zeros(len(self.position))
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def get_n_layers(self):
        return int(round(self.position[0]))

    def update_velocity(self, global_best):
        w, c1, c2 = self.position[1:]
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self):
        self.position += self.velocity
        # Clip hidden layers to [1, 15] and round to integer
        self.position[0] = np.clip(round(self.position[0]), 1, 15)
        # Clip PSO params
        self.position[1:] = np.clip(self.position[1:], 0.1, 2.0)

def run_layer_pso(n_particles=5, max_iter=5):
    swarm = [LayerParticle() for _ in range(n_particles)]
    global_best = None
    global_best_value = float('inf')

    for _ in tqdm(range(max_iter)):
        for p in swarm:
            n_layers = p.get_n_layers()
            best_weights, train_mse = run_architecture_pso(n_layers)

            if train_mse < global_best_value:
                global_best_value = train_mse
                global_best = np.copy(p.position)
                p.best_value = train_mse
                p.best_position = np.copy(p.position)
            p.update_velocity(global_best)
            p.update_position()

    return global_best, global_best_value







# ==============================================
# Main Execution
# ==============================================
if __name__ == "__main__":

    # Run layer optimization
    best_layers, _ = run_layer_pso()
    best_n_layers = int(round(best_layers[0]))

    # Run architecture optimization
    best_arch, _ = run_architecture_pso(best_n_layers)
    best_hidden = best_arch[1:-3].astype(int)

    # Retrain with best architecture
    final_weights, train_mse = run_weight_pso(best_n_layers,best_hidden)

    print(f"\nParameters: {final_weights}")
    print(f"Train MSE: {train_mse:.4f}")