import random
import h2o
import os
import pandas as pd
import numpy as np
from h2o.estimators import H2OEstimator

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

crossover_rate = 0.85
mutation_rate = 0.70


def train(population, generations, train_data, target_col='label'):
    # Convert the training data to an H2O Frame
    train_frame = h2o.H2OFrame(train_data)

    # Specify the target column
    train_frame[target_col] = train_frame[target_col].asfactor()

    # Split train data into training and testing (80/20)
    train_frame, test_frame = train_frame.split_frame(ratios=[0.8])

    # Extract mnist valid labels
    train_labels = train_frame.as_data_frame()[target_col].tolist()

    for generation in range(generations):
        print("Evaluating Population...")

        fitness_scores = []
        for model in population:
            # Train population with train data
            model.train(x=train_frame.columns, y=target_col, training_frame=train_frame)

            # Make predictions on the test data
            predictions = model.predict(test_frame).as_data_frame()['predict'].tolist()

            # Compute accuracy for each model
            accuracy = fitness(predictions, train_labels)
            print("Accuracy", accuracy)
            fitness_scores.append(accuracy)

        # Select the best models for reproduction (elitism)
        [parent0, parent1] = best_chromosomes(population, fitness_scores)

        if np.random.rand() <= crossover_rate:
            print("Crossover...")
            chromosome = crossover(parent0, parent1)

            if np.random.rand() <= mutation_rate:
                print("Mutation...")
                child = mutate(chromosome)
                population.append(child)
            else:
                population.append(chromosome)

            # Remove the least fit model in population
            worse = worse_fit(population, fitness_scores)
            population.remove(worse)

        # Check if current fitness is better than the best fitness so far


def worse_fit(population, fitness_scores):
    score_idx = 0
    score_min = fitness_scores[0]
    for i, score in enumerate(fitness_scores):
        if score < score_min:
            score_min = score
            score_idx = i

    return population[score_idx]


def best_chromosomes(population, fitness_scores):
    total_score = sum(fitness_scores)
    return np.random.choice(population, size=2, replace=True, p=[score / total_score for score in fitness_scores])


def fitness(predictions, train_labels):
    correct_predictions = sum([pred == label for pred, label in zip(predictions, train_labels)])

    return correct_predictions / len(train_labels)


def crossover(p0: H2OEstimator, p1: H2OEstimator):
    if isinstance(p0, H2ORandomForestEstimator) or isinstance(p0, H2OGradientBoostingEstimator):
        ntrees, max_depth = (p0.ntrees, p1.max_depth)\
            if np.random.choice([True, False])\
            else (p1.ntrees, p0.max_depth)

        p0.ntrees = ntrees
        p0.max_depth = max_depth
        return p0
    pass


def mutate(chromosome: H2OEstimator):
    if isinstance(chromosome, H2ORandomForestEstimator) or isinstance(chromosome, H2OGradientBoostingEstimator):
        chromosome.ntrees = np.random.randint(chromosome.ntrees * 0.8, chromosome.ntrees * 1.2)
        chromosome.max_depth = np.random.randint(chromosome.max_depth * 0.8, chromosome.max_depth * 1.2)
        return chromosome

    pass


current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')
train_csv = pd.read_csv(os.path.join(resources_dir, 'train.csv'))

# Start H2O
h2o.init(ip="localhost", port=54321, max_mem_size_GB=4)

population_size = random.randint(2, 5)
generations_size = random.randint(2, 2)

print("Population", population_size)
print("Generations", generations_size)

print("Initializing Distributed Random Forest...")
drf = [H2ORandomForestEstimator(
    ntrees=random.randint(5, 10),
    max_depth=random.randint(3, 7)
) for _ in range(population_size)]

print("Initializing Gradient Boosting Estimator...")
gbm = [H2OGradientBoostingEstimator(
    ntrees=random.randint(5, 10),
    max_depth=random.randint(3, 7)
) for _ in range(population_size)]

train(drf, generations_size, train_csv)
train(gbm, generations_size, train_csv)

print("Training Gradient Boosting....")
#train(gbm, generations_size, train_csv)

print("Training Deep Learning")
train(deep_learning, generations_size, train_csv)

print("DONE")
while True:
    pass
