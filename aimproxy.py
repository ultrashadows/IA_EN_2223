import random
import time

import h2o
import os
import pandas as pd
import numpy as np

from h2o.estimators import H2OEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

crossover_rate = 0.85
mutation_rate = 0.70


def test(best_model, train_data, test_data, target_col='label'):
    train_frame = h2o.H2OFrame(train_data)
    train_frame[target_col] = train_frame[target_col].asfactor()

    test_frame = h2o.H2OFrame(test_data)
    test_frame[target_col] = test_frame[target_col].asfactor()

    print("Training best model...")
    best_model.train(x=train_frame.columns, y=target_col, training_frame=train_frame)

    print("Predicting best model...")
    predictions = best_model.predict(test_frame).as_data_frame()['predict'].tolist()

    print("Creating Kaggle frame...")
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

    for generation in range(generations):
        print("Evaluating Population...")

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

        # TODO 1 Check if current fitness is better than the best fitness so far
    # TODO 2 Save the best model


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


current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')
train_csv = pd.read_csv(os.path.join(resources_dir, 'train.csv'))

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

print("Initializing Gradient Boosting...")
gbm = [H2OGradientBoostingEstimator(
    ntrees=random.randint(5, 10),
    max_depth=random.randint(3, 7)
) for _ in range(population_size)]

print("Initializing Deep Learning")
deep_learning = [H2ODeepLearningEstimator(
    hidden=[25, 50, 40],
    epochs=5
) for _ in range(population_size)]

print("Training Distributed Random Forest...")
# train(drf, generations_size, train_csv)

print("Training Gradient Boosting....")
# train(gbm, generations_size, train_csv)

print("Training Deep Learning")
train(deep_learning, generations_size, train_csv)

kaggle = test(best_model=None, train_data=train_csv, test_data=None)
current_time_millis = int(round(time.time() * 1000))
results_dir = os.path.join(current_dir, 'predictions_' + str(current_time_millis) + ".csv")
kaggle.to_csv(results_dir, index=False)

# TODO 3 Load model into population
# TODO 4 Plot Accuracy across generations
