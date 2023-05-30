# Import necessary libraries
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2ODeepLearningEstimator
import pandas as pd
import numpy as np
import os
import time
import logging

# Start H2O
h2o.init(ip="localhost", port=54321)

# Change default H2O settings for performance
h2o.no_progress()

# Fetch resources directory
current_dir = os.path.dirname(__file__)
resources_dir = os.path.join(current_dir, 'resources')
results_dir = os.path.join(current_dir, 'results')
logs_dir = os.path.join(current_dir, 'logs')

# Create directories if they do not exist
for dir in [resources_dir, results_dir, logs_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Logger config
current_time_millis = int(round(time.time() * 1000))
log_file = os.path.join(logs_dir, 'log_' + str(current_time_millis) + ".log")
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%d:%m:%Y %H:%M:%S')

# Load train data from Kaggle competition
logging.info("Importing and handling training and testing data...")
train_data = pd.read_csv(os.path.join(resources_dir, 'train.csv'))

# Split train data into training and testing (80/20)
train_data_len = len(train_data)
train_data_split = int(train_data_len * 0.8)
train_data_train = train_data.iloc[:train_data_split]
train_data_test = train_data.iloc[train_data_split:]

# Convert data to H2O frames
train_h2o_train = h2o.H2OFrame(train_data_train)
train_h2o_test = h2o.H2OFrame(train_data_test)

# Load test data
test_data = pd.read_csv(os.path.join(resources_dir, 'test.csv'))
test_h2o = h2o.H2OFrame(test_data)

# Set the response column for training purposes
train_h2o_train["label"] = train_h2o_train["label"].asfactor()

# Define three base models
rf_base = H2ORandomForestEstimator(ntrees=20, max_depth=10, nfolds=5, seed=123, max_runtime_secs=300)
gbm_base = H2OGradientBoostingEstimator(ntrees=20, max_depth=10, nfolds=5, seed=123, max_runtime_secs=300)
dl_base = H2ODeepLearningEstimator(hidden=[50, 50], epochs=10, nfolds=5, seed=123, max_runtime_secs=300)

# Add base models to a list
base_models = [rf_base, gbm_base, dl_base]

# Define genetic algorithm parameters
# Currently using default values recommended by past studies
pop_size = 10
num_generations = 1
mutation_rate = 0.2
crossover_rate = 0.85

models_trained = 0


# Define fitness function
def fitness(model):
    global models_trained
    models_trained += 1

    model.train(x=train_h2o_train.names[:-1], y="label", training_frame=train_h2o_train)
    predictions = model.predict(train_h2o_test).as_data_frame()['predict'].tolist()
    label_guess = pd.Series(train_data_train['label']).mode()[0]
    predictions = [label_guess if np.isnan(x) else x for x in predictions]
    correct_predictions = sum([pred == label for pred, label in zip(predictions, train_data_test['label'])])
    accuracy = correct_predictions / len(train_data_test)
    print("Model Accuracy: {:.2f}%".format(accuracy * 100), " , Models Trained: ", models_trained, "/",
          str(pop_size * num_generations))
    logging.info("Model Accuracy: {:.2f}%".format(accuracy * 100) + " , Models Trained: "
                 + str(models_trained) + "/" + str(pop_size * num_generations))
    return accuracy, correct_predictions, predictions


# Define genetic operators (crossover and mutation)
def crossover(model1, model2):
    model1_params = model1.params
    model2_params = model2.params

    for key in model1_params:
        if isinstance(model1_params[key], list):
            if np.random.rand() > 0.5:
                model1_params[key], model2_params[key] = model2_params[key], model1_params[key]

    new_model = model1.__class__()
    new_model._parms = model1_params
    return new_model


def mutation(model):
    model_params = model.params

    for key in model_params:
        if isinstance(model_params[key], list):
            if np.random.rand() > 0.5:
                new_val = np.random.choice(model_params[key])
                model_params[key] = new_val

    new_model = model.__class__()
    new_model._parms = model_params
    return new_model


# Initialize population
population = [np.random.choice(base_models) for _ in range(pop_size)]

# Initialize best accuracy
best_accuracy = 0

# Initialize best model
best_model_file = os.path.join(resources_dir, 'best_model')
if os.path.exists(best_model_file):
    # Load the saved model
    best_model = h2o.load_model(best_model_file)
else:
    # No best model found, create a new one
    best_model = None

# Evolve population
for generation in range(num_generations):
    logging.info("Beginning models training...")

    # Evaluate fitness
    fitness_scores = [fitness(model) for model in population]

    # Select parents for crossover
    parent1, parent2 = np.random.choice(population, size=2, replace=True,
                                        p=[score[0] for score in fitness_scores] / np.sum(
                                            [score[0] for score in fitness_scores]))

    # Perform genetic operations
    child1 = crossover(parent1, parent2)
    child2 = mutation(child1)

    # Replace least fit model in population with new child
    least_fit = np.argmin([score[0] for score in fitness_scores])
    population[least_fit] = child2

    # Evaluate best fitness and model
    current_accuracy = max([score[0] for score in fitness_scores])
    current_model = population[np.argmax([score[0] for score in fitness_scores])]

    # Check if current fitness is better than the best fitness so far
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = current_model

    # Print progress
    print("Generation: ", generation, " Best accuracy: ", max([score[0] for score in fitness_scores]))
    logging.info("Generation training complete!")
    logging.info("Generation: " + str(generation) + " Best accuracy: "
                 + str(max([score[0] for score in fitness_scores])))

# Train best model on full training data
logging.info("Best model found, training best model...")
best_model.train(x=train_h2o_train.names[:-1], y="label", training_frame=train_h2o_train)

# Make predictions on the test data
logging.info("Predicting test data labels...")
best_model_predictions = best_model.predict(test_h2o).as_data_frame()['predict'].tolist()

# Save the best model
logging.info("Saving the best model for future training...")
h2o.save_model(best_model, path=os.path.join(resources_dir, 'best_model'), force=True)

# Create result.csv containing the ID of the test.csv line and the predicted number ("label")
result_df = pd.DataFrame({'ID': test_data['Id'], 'label': best_model_predictions})

# Save result to results folder
logging.info("Saving results...")
result_df.to_csv(os.path.join(results_dir, 'predictions_' + str(current_time_millis) + ".csv"), index=False)
logging.info("Run done!")
