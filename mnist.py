# Import necessary libraries
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator
import pandas as pd
import numpy as np
import os

# Start H2O
h2o.init(ip="localhost", port=54321)

# Fetch resources directory
current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')

# Load train and test data from Kaggle competition
train_data = pd.read_csv(os.path.join(resources_dir, 'train.csv'))
test_data = pd.read_csv(os.path.join(resources_dir, 'test.csv'))

# Convert data to H2O frames
train_h2o = h2o.H2OFrame(train_data)
test_h2o = h2o.H2OFrame(test_data)

# Set the response column for training purposes
train_h2o["label"] = train_h2o["label"].asfactor()

# Define three base models
rf_base = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=5)
gbm_base = H2OGradientBoostingEstimator(ntrees=50, max_depth=5, nfolds=5)
dl_base = H2ODeepLearningEstimator(hidden=[100, 100, 100], epochs=10, nfolds=5)

# Add base models to a list
base_models = [rf_base, gbm_base, dl_base]

# Define genetic algorithm parameters
# Currently using default values recommended by past studies
pop_size = 100
num_generations = 100
mutation_rate = 0.2
crossover_rate = 0.85


# Define fitness function
def fitness(model):
    model.train(x=train_h2o.names[:-1], y="label", training_frame=train_h2o)
    predictions = model.predict(test_h2o).as_data_frame()['predict'].tolist()
    label_guess = pd.Series(train_data['label']).mode()[0]
    predictions = [label_guess if np.isnan(x) else x for x in predictions]
    accuracy = sum(predictions == train_data['label']) / len(train_data)
    return accuracy


# Define genetic operators (crossover and mutation)
def crossover(model1, model2):
    model1_params = model1.params
    model2_params = model2.params

    for key in model1_params:
        if isinstance(model1_params[key], list):
            if np.random.rand() > 0.5:
                model1_params[key], model2_params[key] = model2_params[key], model1_params[key]

    new_model = model1.__class__(**model1_params)
    return new_model


def mutation(model):
    model_params = model.params

    for key in model_params:
        if isinstance(model_params[key], list):
            if np.random.rand() > 0.5:
                new_val = np.random.choice(model_params[key])
                model_params[key] = new_val

    new_model = model.__class__(**model_params)
    return new_model


# Initialize population
population = [np.random.choice(base_models) for _ in range(pop_size)]

best_fitness = 0
best_model = None
# Evolve population
for generation in range(num_generations):

    # Evaluate fitness
    fitness_scores = [fitness(model) for model in population]

    # Select parents for crossover
    parent1, parent2 = np.random.choice(population, size=2, replace=False, p=fitness_scores/np.sum(fitness_scores))

    # Perform genetic operations
    child1 = crossover(parent1, parent2)
    child2 = mutation(child1)

    # Replace least fit model in population with new child
    least_fit = np.argmin(fitness_scores)
    population[least_fit] = child2

    # Evaluate best fitness and model
    current_fitness = max(fitness_scores)
    current_model = population[np.argmax(fitness_scores)]

    # Check if current fitness is better than the best fitness so far
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_model = current_model

    # Print progress
    print("Generation: ", generation, " Best fitness: ", max(fitness_scores))


# Evaluate best model on test set
best_model = max(population, key=fitness)
best_model.train(x=train_h2o.names[:-1], y="label", training_frame=train_h2o)
test_accuracy = best_model.model_performance(test_data=test_h2o).accuracy()
print("Test Accuracy: ", test_accuracy)
