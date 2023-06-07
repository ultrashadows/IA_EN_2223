import random
import time

import h2o
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from h2o.estimators import H2OEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

# ClosedAI crossover and mutation params
crossover_rate = 0.90
mutation_rate = 0.90

# OS Load datasets for train and test data
current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')
train_csv = pd.read_csv(os.path.join(resources_dir, 'train.csv'))
test_csv = pd.read_csv(os.path.join(resources_dir, 'test.csv'))

# H2O Init
h2o.init(ip="localhost", port=54321, max_mem_size_GB=2)


def test(model, train_data, test_data, target_col='label'):
    train_frame = h2o.H2OFrame(train_data)
    train_frame[target_col] = train_frame[target_col].asfactor()

    test_frame = h2o.H2OFrame(test_data)

    model.train(x=train_frame.columns, y=target_col, training_frame=train_frame)
    predictions = model.predict(test_frame).as_data_frame()['predict'].tolist()

    return pd.DataFrame({'ID': test_data['Id'], 'label': predictions})


def train(population, generations, train_data, target_col='label'):
    # Convert the training data to an H2O Frame
    train_frame = h2o.H2OFrame(train_data)

    # Specify the target column
    train_frame[target_col] = train_frame[target_col].asfactor()

    # Split train data into training and testing (80/20)
    train_frame, test_frame = train_frame.split_frame(ratios=[0.8])

    # Extract mnist valid labels
    test_labels = test_frame.as_data_frame()[target_col].tolist()

    # Variables to define the best generation model
    accuracy_across_generations = []
    generation_accuracy = 0
    generation_model = None
    for generation in range(generations):
        print("Generation", generation)

        # Run Train and Test on train & test data frames
        fitness_scores = evolve(population, train_frame, test_frame, target_col, test_labels)

        # Variables to define the best population model
        population_accuracy = max(fitness_scores)
        population_model = population[fitness_scores.index(population_accuracy)]
        accuracy_across_generations.append(population_accuracy)

        # Select the best models for reproduction (elitism)
        [parent0, parent1] = best_chromosomes(population, fitness_scores)

        # Perform genetic operations
        if np.random.rand() <= crossover_rate:
            chromosome = crossover(parent0, parent1)

            if np.random.rand() <= mutation_rate:
                child = mutate(chromosome)
                population.append(child)
            else:
                population.append(chromosome)

            # Remove the worst fit model in population
            worse = worst_fit(population, fitness_scores)
            population.remove(worse)

        # Check if current fitness is better than the best fitness so far
        if population_accuracy > generation_accuracy:
            generation_accuracy = population_accuracy
            generation_model = population_model

    return generation_model, generation_accuracy, accuracy_across_generations


def evolve(population, train_frame, test_frame, target_col, test_labels):
    fitness_scores = []
    for model in population:
        # Train population with train data
        model.train(x=train_frame.columns, y=target_col, training_frame=train_frame)

        # Make predictions on the test data
        predictions = model.predict(test_frame).as_data_frame()['predict'].tolist()

        # Compute accuracy for each model
        accuracy = fitness(predictions, test_labels)

        print("Accuracy", accuracy)
        fitness_scores.append(accuracy)

    return fitness_scores


def worst_fit(population, fitness_scores):
    worst_score = min(fitness_scores)
    score_idx = fitness_scores.index(worst_score)
    return population[score_idx]


def best_chromosomes(population, fitness_scores):
    total_score = sum(fitness_scores)
    probability = [score / total_score for score in fitness_scores]
    return np.random.choice(population, size=2, replace=True, p=probability)


def fitness(predictions, train_labels):
    correct_predictions = sum([pred == label for pred, label in zip(predictions, train_labels)])
    return correct_predictions / len(train_labels)


def crossover(p0: H2OEstimator, p1: H2OEstimator):
    if isinstance(p0, H2ORandomForestEstimator) or isinstance(p0, H2OGradientBoostingEstimator):
        ntrees, max_depth = (p0.ntrees, p1.max_depth) \
            if np.random.choice([True, False]) \
            else (p1.ntrees, p0.max_depth)

        p0.ntrees = ntrees
        p0.max_depth = max_depth
        return p0

    if isinstance(p0, H2ODeepLearningEstimator):
        p0.hidden[1] = p1.hidden[1]
        return p0
    pass


def mutate(chromosome: H2OEstimator):
    if isinstance(chromosome, H2ORandomForestEstimator) or isinstance(chromosome, H2OGradientBoostingEstimator):
        chromosome.ntrees = np.random.randint(chromosome.ntrees * 0.8, chromosome.ntrees * 1.2)
        chromosome.max_depth = np.random.randint(chromosome.max_depth * 0.8, chromosome.max_depth * 1.2)
        return chromosome

    if isinstance(chromosome, H2ODeepLearningEstimator):
        layer0 = np.random.randint(chromosome.hidden[0] * 0.85, chromosome.hidden[0] * 1.15)
        layer1 = np.random.randint(chromosome.hidden[1] * 0.95, chromosome.hidden[1] * 1.25)
        layer2 = np.random.randint(chromosome.hidden[2] * 0.75, chromosome.hidden[2] * 1.05)

        chromosome.hidden = [layer0, layer1, layer2]
        chromosome.epochs = np.random.randint(chromosome.epochs * 0.85, chromosome.epochs * 1.15)
        return chromosome
    pass


def accuracy_across_generations_graph(graph_dir, generations, accuracy0, accuracy1, accuracy3):
    plt.plot(generations, accuracy0, marker='o', linestyle='-', color='b', label='Random Forest')
    plt.plot(generations, accuracy1, marker='o', linestyle='-', color='g', label='Gradient Boosting')
    plt.plot(generations, accuracy3, marker='o', linestyle='-', color='r', label='Deep Learning')
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of Accuracy across {len(generations)} generations')
    plt.legend()
    plt.grid(True)

    graph_file = os.path.join(graph_dir, 'accuracy_across_generations_graph.png')
    plt.savefig(graph_file)


"""
ClosedAI MNIST Problem (DRF, GBM, DL)
"""
population_size = random.randint(2, 2)
generations_size = random.randint(2, 2)
print("Population, Generations", population_size, generations_size)

print("Initializing Distributed Random Forest...")
drf_population = [H2ORandomForestEstimator(
    ntrees=random.randint(5, 10),
    max_depth=random.randint(3, 7)
) for _ in range(population_size)]

print("Initializing Gradient Boosting...")
gbm_population = [H2OGradientBoostingEstimator(
    ntrees=random.randint(5, 10),
    max_depth=random.randint(3, 7)
) for _ in range(population_size)]

print("Initializing Deep Learning...")
dl_population = [H2ODeepLearningEstimator(
    hidden=[25, 50, 40],
    epochs=5
) for _ in range(population_size)]

# Variable to generate files based on timestamp
timestamp = int(round(time.time() * 1000))

"""
Distributed Random Forest
"""
print("Training DRF...")
drf_model, drf_accuracy, drf_accuracies = train(drf_population, generations_size, train_csv)
h2o.save_model(drf_model, path=resources_dir, filename=f'drf_{str(timestamp)}', force=True)

"""
Gradient Boosting
"""
print("Training GBM....")
gbm_model, gdm_accuracy, gdm_accuracies = train(gbm_population, generations_size, train_csv)
h2o.save_model(gbm_model, path=resources_dir, filename=f'gbm_{str(timestamp)}', force=True)

"""
Deep Learning
"""
print("Training DL...")
dl_model, dl_accuracy, dl_accuracies = train(dl_population, generations_size, train_csv)
h2o.save_model(dl_model, path=resources_dir, filename=f'dl_{str(timestamp)}.txt', force=True)

"""
Accuracy Graph
"""
accuracy_across_generations_graph(
    resources_dir,
    range(generations_size),
    drf_accuracies,
    gdm_accuracies,
    dl_accuracies
)

"""
Kaggle
"""
least_models = [drf_accuracy, gdm_accuracy, dl_accuracy]
best_accuracy = max(least_models)
best_model = least_models[least_models.index(best_accuracy)]

kaggle_file = os.path.join(resources_dir, f'kaggle_{str(timestamp)}.csv')
dl_kaggle = test(dl_model, train_csv, test_csv)
dl_kaggle.to_csv(kaggle_file, index=False)

# TODO 3 Load model into population
