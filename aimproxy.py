import random
import h2o
import os
import pandas as pd

from h2o.estimators.random_forest import H2ORandomForestEstimator


def train(train_data, target_col='label'):
    # Convert the training data to an H2O Frame
    train_frame = h2o.H2OFrame(train_data)

    # Specify the target column
    train_frame[target_col] = train_frame[target_col].asfactor()

    # Split train data into training and testing (80/20)
    train_frame, test_frame = train_frame.split_frame(ratios=[0.8])

    population = initialize_population()

    generations_size = random.randint(2, 20)
    for generation in range(generations_size):
        fitness_scores = evaluate_population(population, train_frame, test_frame, target_col)

        # Select the best models for reproduction (elitism)
        [parent0, parent1] = best_chromosomes(fitness_scores)

        chromosome = crossover(parent0, parent1)
        # child = mutate(chromosome)

        # Replace the least fit model in population with new child

        # Check if current fitness is better than the best fitness so far


def initialize_population():
    # Initialize the population
    population = []
    population_size = random.randint(2, 5)

    for _ in range(population_size):
        model = H2ORandomForestEstimator(
            ntrees=random.randint(5, 10),
            max_depth=random.randint(5, 10),
            min_rows=random.randint(1, 5),
        )
        population.append(model)

    return population


def evaluate_population(population, train_frame, test_frame, target_col):
    fitness_scores = []
    for model in population:
        # Train population with train data
        model.train(x=train_frame.columns, y=target_col, training_frame=train_frame)

        # Make predictions on the test data
        predictions = model.predict(test_frame)

        # Compute accuracy for each model
        accuracy = fitness(predictions, train_frame)
        fitness_scores.append((model, accuracy))

    return fitness_scores


def fitness(predictions, train_frame):
    correct_predictions = sum([pred == label for pred, label in zip(predictions, train_frame['label'])])

    return correct_predictions / len(train_labels)



def best_chromosomes(fitness_scores):
    # Sort the population based on fitness scores in descending order
    fitness_scores.sort(key=lambda x: x[1], reverse=True)

    return [model for model, _ in fitness_scores[:2]]


def crossover(parent0, parent1):
    # Perform crossover operation to create a new child model
    child = H2ORandomForestEstimator(
        ntrees=random.randint(parent0.ntrees, parent1.ntrees),
        max_depth=random.randint(parent0.max_depth, parent1.max_depth),
        min_rows=random.randint(parent0.min_rows, parent1.min_rows),
    )

    return child


def mutate(chromosome):
    pass


current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')
train_csv = pd.read_csv(os.path.join(resources_dir, 'train.csv'))

# Start H2O
h2o.init(ip="localhost", port=54321, max_mem_size_GB=2)
h2o.no_progress()

train(train_csv)
