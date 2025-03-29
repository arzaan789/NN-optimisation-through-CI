from lib import modified_carpriceprediction
import numpy as np
import copy
import random
from tqdm import tqdm

evaluate_train = modified_carpriceprediction.modified_make_evaluator('train')
evaluate_val = modified_carpriceprediction.modified_make_evaluator('validation')
evaluate_test = modified_carpriceprediction.modified_make_evaluator('test')

class Neuron:
    def __init__(self, bias=None, weights=None):
        self.bias = bias  # The bias term for the neuron
        self.weights = weights  # The weights connecting to previous layer neurons

class Layer:
    def __init__(self, num_neurons, prev_layer_size=None):
        self.neurons = []  # List of Neuron objects in this layer
        self.num_neurons = num_neurons  # Number of neurons in this layer
        for _ in range(num_neurons):
            bias = np.random.randn()  # Randomly initializing the bias
            weights = np.random.randn(prev_layer_size) if prev_layer_size else None  # Randomly initializing the weights
            self.neurons.append(Neuron(bias=bias, weights=weights))


class NeuralNetwork:
    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = []
        self.layer_sizes = layer_sizes
        self.layers = []  # List of Layer objects
        for i in range(1, len(layer_sizes)):
            prev_layer_size = layer_sizes[i - 1]
            layer_size = layer_sizes[i]
            self.layers.append(Layer(layer_size, prev_layer_size))


    def build_from_layers(self,layers):
        self.layers = layers
        self.layer_sizes = [21] + [layer.num_neurons for layer in layers]



    def __str__(self):
        representation = ""
        full_network = [21] + [layer.num_neurons for layer in self.layers]
        representation += f"{full_network}"
        return representation

    def get_layer_array(self):
        return [21] + [layer.num_neurons for layer in self.layers]

    def full_str(self):
        representation = ""
        full_network = [21] + [layer.num_neurons for layer in self.layers]
        representation += f"{full_network}\n\n"
        for i, layer in enumerate(self.layers):
            representation += f"Layer {i}:\n"
            for j, neuron in enumerate(layer.neurons):
                representation += f"Neuron {j}:\n"
                representation += f"Bias: {neuron.bias}\n"
                representation += f"Weights: {neuron.weights}\n"
        return representation


    def add_layer(self):
        if len(self.layers) >= 10:  # Ensure we don't exceed max layers
            return

            # Select a random position (not before input or after output layer)
        layer_idx = random.randint(0, len(self.layers) - 1)

        # Get neighboring layers
        next_layer = self.layers[layer_idx]

        # Define the number of neurons for the new layer
        new_neuron_count = random.randint(2, 20)
        # Create the new layer
        prev_neurons = self.layers[layer_idx - 1].num_neurons if layer_idx > 0 else self.layer_sizes[0]
        new_layer = Layer(new_neuron_count,prev_neurons)
        self.layers[layer_idx] = Layer(next_layer.num_neurons, new_neuron_count)

        # Insert the new layer into the network
        self.layers.insert(layer_idx, new_layer)

        return

    def remove_layer(self):
        if len(self.layers) <= 2:  # Ensure we always have input and output layers
            return

        # Select a random hidden layer to remove (not the input or output layer)
        layer_idx = random.randint(0, len(self.layers) - 2)


        next_layer = self.layers[layer_idx + 1]

        # Rewire: Set next layer's weights to connect with previous layer
        prev_neurons = self.layers[layer_idx - 1].num_neurons if layer_idx > 0 else self.layer_sizes[0]
        self.layers[layer_idx+1] = Layer(next_layer.num_neurons, prev_neurons)

        # Remove the layer
        self.layers.pop(layer_idx)

        return

    def add_neurons(self):
        # Select a random layer (excluding input and output layers)
        if len(self.layers) <= 2:  # Ensure we don't add neurons to just input-output layers
            return self

        # Randomly pick a hidden layer
        layer_idx = random.randint(0, len(self.layers) - 2)  # Exclude input and output layers
        layer = self.layers[layer_idx]

        # Randomly choose the number of neurons to add (between 1 and 3)
        new_neurons = random.randint(1, 3)

        # Add the new neurons to the layer
        old_neuron_count = layer.num_neurons
        layer.num_neurons += new_neurons  # Update the number of neurons in this layer

        # Initialize new neurons
        for i in range(new_neurons):
            bias = np.random.randn()  # Randomly initializing the bias
            prev_neurons = self.layers[layer_idx - 1].num_neurons if layer_idx > 0 else self.layer_sizes[0]
            weights = np.random.randn(prev_neurons)
            self.layers[layer_idx].neurons.append(Neuron(bias=bias, weights=weights))

        # Update the weights of the next layer
        for i in range(len(self.layers[layer_idx + 1].neurons)):
            self.layers[layer_idx + 1].neurons[i].weights = np.append(
                self.layers[layer_idx + 1].neurons[i].weights,
                np.random.randn(new_neurons)
            )

        return self

    def remove_neuron(self):
        # Select a random layer (excluding input and output layers)
        if len(self.layers) <= 2:  # Ensure we don't remove neurons from input or output layers
            return self

        # Randomly pick a hidden layer
        layer_idx = random.randint(0, len(self.layers) - 2)  # Exclude input and output layers
        layer = self.layers[layer_idx]

        # If the layer has more than one neuron, remove one randomly
        if len(layer.neurons) > 1:
            neuron_idx = random.randint(0, len(layer.neurons) - 1)  # Randomly pick a neuron to remove
            # Remove the selected neuron
            layer.neurons.pop(neuron_idx)
            layer.num_neurons -= 1  # Decrease the number of neurons in the layer

            next_layer = self.layers[layer_idx + 1]
            # Remove the corresponding weights from the next layer
            for neuron in next_layer.neurons:
                neuron.weights = np.delete(neuron.weights, neuron_idx)

        return self

    def swap_layers(self):
        # Select two random layers (excluding the input and output layers)
        if len(self.layers) <= 2:  # Ensure we have hidden layers to swap
            return self

        # Randomly pick two distinct hidden layers
        layer_idx1 = random.randint(0, len(self.layers) - 2)
        layer_idx2 = random.randint(0, len(self.layers) - 2)

        # Ensure the two selected layers are distinct
        while layer_idx1 == layer_idx2:
            layer_idx2 = random.randint(0, len(self.layers) - 2)

        # Store the weights and biases of the layers
        layer1 = self.layers[layer_idx1]
        layer2 = self.layers[layer_idx2]

        if layer_idx1 < layer_idx2:

            prev_neurons = self.layers[layer_idx1 - 1].num_neurons if layer_idx1 > 0 else self.layer_sizes[0]
            self.layers[layer_idx1] = Layer(layer2.num_neurons, prev_neurons)

            if abs(layer_idx1 - layer_idx2) == 1: # if they are together
                self.layers[layer_idx2] = Layer(layer1.num_neurons, layer2.num_neurons)
            else:
                self.layers[layer_idx1+1] = Layer(self.layers[layer_idx1+1].num_neurons, self.layers[layer_idx1].num_neurons)
                self.layers[layer_idx2] = Layer(layer1.num_neurons, self.layers[layer_idx2 - 1].num_neurons)

            self.layers[layer_idx2 + 1] = Layer(self.layers[layer_idx2 + 1].num_neurons, self.layers[layer_idx2].num_neurons)

        else:
            prev_neurons = self.layers[layer_idx2 - 1].num_neurons if layer_idx2 > 0 else self.layer_sizes[0]
            self.layers[layer_idx2] = Layer(layer1.num_neurons, prev_neurons)

            if abs(layer_idx1 - layer_idx2) == 1:
                self.layers[layer_idx1] = Layer(layer2.num_neurons, layer1.num_neurons)
            else:
                self.layers[layer_idx2+1] = Layer(self.layers[layer_idx2+1].num_neurons, self.layers[layer_idx2].num_neurons)
                self.layers[layer_idx1] = Layer(layer2.num_neurons, self.layers[layer_idx1 - 1].num_neurons)

            self.layers[layer_idx1 + 1] = Layer(self.layers[layer_idx1 + 1].num_neurons, self.layers[layer_idx1].num_neurons)

        return self

    def mutate_weights_gaussian(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights += np.random.normal(0, 0.01, size=len(neuron.weights))
                neuron.bias += np.random.normal(0, 0.01)  # Mutate bias too
        return self

    def weights_swap_between_neurons(self):
        for i in range(len(self.layers)):
            if len(self.layers[i].neurons) > 1:  # Ensure at least two neurons exist
                idx1, idx2 = np.random.choice(len(self.layers[i].neurons), 2, replace=False)
                self.layers[i].neurons[idx1].weights, self.layers[i].neurons[idx2].weights = self.layers[i].neurons[idx2].weights.copy(), self.layers[i].neurons[idx1].weights.copy()
        return self

    def replace_layer(self, layer_idx, neurons, prev_neurons):
        self.layers[layer_idx] = Layer(neurons, prev_neurons)




class Genetic_algo:

    def initialize_population(self):

        for i in range(self.population_size):
            # make random array of size randomly between 1 and 50
            length = np.random.randint(1, 15)
            list = [0 for i in range(length)]
            #fill randomly with values between 1 and 25
            for i in range(length):
                list[i] = np.random.randint(1, 25)

            list.insert(0, 21)
            list.append(1)

            network = NeuralNetwork(list)
            self.population.append(network)

    def mutate(self, network, current_iteration):
        #make a copy of the network
        network_copy = copy.deepcopy(network)
        # decrease mutation rate over time from 0.5 to 0.01
        structural_mutation_rate = 0.5 - 0.49 * current_iteration / self.iterations
        # increase mutation rate over time from 0.01 to 0.3
        weight_mutation_rate = 0.01 + 0.49 * current_iteration / self.iterations
        if random.random() < structural_mutation_rate:
            mutation = random.randint(0, 4)
            if mutation == 0:
                network_copy.add_layer()
            elif mutation == 1:
                network_copy.remove_layer()
            elif mutation == 2:
                network_copy.add_neurons()
            elif mutation == 3:
                network_copy.remove_neuron()
            else:
                network_copy.swap_layers()

        if random.random() < weight_mutation_rate:
            mutation = random.randint(0, 1)
            if mutation == 0:
                network_copy.mutate_weights_gaussian()
            else:
                network_copy.weights_swap_between_neurons()

        return network_copy

    def layer_crossover(self, parent1, parent2):

        # randomly select a layer from parent 1
        p1_layer_idx = random.randint(0, len(parent1.layers) - 2)
        # randomly select a layer from parent 2
        p2_layer_idx = random.randint(0, len(parent2.layers) - 2)

        p1_layer_num_neurons = parent1.layers[p1_layer_idx].num_neurons
        p2_layer_num_neurons = parent2.layers[p2_layer_idx].num_neurons

        prev_neurons = parent1.layers[p1_layer_idx - 1].num_neurons if p1_layer_idx > 0 else parent1.layer_sizes[0]
        parent1.replace_layer(p1_layer_idx, p2_layer_num_neurons, prev_neurons)
        parent1.layers[p1_layer_idx + 1] = Layer(parent1.layers[p1_layer_idx + 1].num_neurons, p2_layer_num_neurons)


        prev_neurons = parent2.layers[p2_layer_idx - 1].num_neurons if p2_layer_idx > 0 else parent2.layer_sizes[0]
        parent2.replace_layer(p2_layer_idx, p1_layer_num_neurons, prev_neurons)
        parent2.layers[p2_layer_idx + 1] = Layer(parent2.layers[p2_layer_idx + 1].num_neurons, p1_layer_num_neurons)

        return parent1, parent2

    def half_crossover(self, parent1, parent2):
        # get length of layers
        length1 = len(parent1.layers)
        parent1_first_half = parent1.layers[:length1//2]
        length2 = len(parent2.layers)
        parent2_second_half = parent2.layers[length2//2:]
        parent1.layers = parent1.layers[:length1//2]
        parent2_second_half[0] = Layer(parent2_second_half[0].num_neurons, parent1.layers[-1].num_neurons)

        layers = parent1.layers + parent2_second_half
        child = NeuralNetwork()
        child.build_from_layers(layers)
        return child

    def weight_crossover(self, parent1, parent2):
        if parent1.get_layer_array() == parent2.get_layer_array():
            child = copy.deepcopy(parent1)
            for i in range(len(child.layers)):
                for j in range(len(child.layers[i].neurons)):
                    child.layers[i].neurons[j].weights = (parent1.layers[i].neurons[j].weights + parent2.layers[i].neurons[j].weights) / 2
                    child.layers[i].neurons[j].bias = (parent1.layers[i].neurons[j].bias + parent2.layers[i].neurons[j].bias) / 2
            return child
        else:
            return parent1, parent2


    def crossover(self, parent1, parent2, current_iteration):
        # structural_crossover_rate from 0.7 to 0.01
        structural_crossover_rate = 0.7 - 0.69 * current_iteration / self.iterations
        # weight_crossover_rate from 0.01 to 0.7
        weight_crossover_rate = 0.01 + 0.69 * current_iteration / self.iterations
        if random.random() < structural_crossover_rate:
            crossover = random.randint(0, 1)
            if crossover == 0:
                return [self.layer_crossover(copy.deepcopy(parent1), copy.deepcopy(parent2))]
            else:
                return [self.half_crossover(copy.deepcopy(parent1), copy.deepcopy(parent2))]

        if random.random() < weight_crossover_rate:
            return [self.weight_crossover(copy.deepcopy(parent1), copy.deepcopy(parent2))]

        return [copy.deepcopy(parent1)]

    def run(self, population_size, iterations):
        convergence_point = 0 # Number of iterations since the last change in the best individual
        self.population_size = population_size
        self.iterations = iterations
        self.population = []

        self.initialize_population()

        best_individual = None
        best_fitness = float('inf')

        for generation in range(iterations):
            fitness_values = [evaluate_train(network) for network in self.population]

            # Sort the population based on fitness values (lower is better)
            sorted_population =sorted(zip(self.population, fitness_values), key=lambda x: x[1])
            best_individual_so_far = sorted_population[0][0]
            best_fitness_so_far = sorted_population[0][1]
            if best_fitness_so_far < best_fitness:
                best_individual = best_individual_so_far
                best_fitness = best_fitness_so_far
                convergence_point = generation

            # Select top 50% of the population
            num_parents = max(1, int(0.5 * self.population_size))
            parents = [net for net, fit in sorted_population[:num_parents]]
            new_population = []
            while len(new_population) < self.population_size:
                if len(parents) >= 2:
                    parent1, parent2 = random.sample(parents, 2)
                else:
                    parent1 = parent2 = parents[0]
                children = self.crossover(parent1, parent2, generation)
                if type(children[0]) == tuple:
                    children = children[0]
                for child in children:
                    child = self.mutate(child, generation)
                    new_population.append(child)
                    if len(new_population) >= self.population_size:
                        break

            self.population = new_population

        return best_individual, convergence_point
