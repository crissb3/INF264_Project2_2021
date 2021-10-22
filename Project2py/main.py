import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time


# function that converts values in images into 1 or 0 depending on if value is over or under 255/2
def one_zero(images_in):
    for row in images_in:
        for value, col in enumerate(row):
            if row[value] > 255/2:
                row[value] = 1
            else:
                row[value] = 0
    return images_in


# plot different max depths in the decision tree classifier to see different accuracy scores
def plot_dt_depth(X, y, param1, param2, param3):
    max_depth_values = {'max_depth': [5, 13, 20, 40, 80]}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=seed, criterion=param1, min_samples_leaf=param2,
                                               min_samples_split=param3),
                        max_depth_values, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    mean_accuracy = grid.cv_results_['mean_test_score']
    plt.plot([5, 13, 20, 40, 80], mean_accuracy)
    plt.xlabel('max_depth value')
    plt.ylabel('Accuracy')
    plt.show()

    
# plot different max depths in the knn classifier for different k values
def plot_knn_n(X, y, param1, param2):
    n_neighbors = {'n_neighbors': [1, 5, 10, 20, 40]}
    grid = GridSearchCV(KNeighborsClassifier(weights=param1, metric=param2),
                        n_neighbors, cv=2, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y.ravel())
    mean_accuracy = grid.cv_results_['mean_test_score']
    plt.plot([1, 5, 10, 20, 40], mean_accuracy)
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.show()


# plot different max depths in the knn classifier for different alpha values
def plot_nn_alpha(X, y, param1, param2):
    alpha = {'alpha': [0.0001, 0.0002, 0.0003, 0.0010]}
    grid = GridSearchCV(MLPClassifier(activation=param1, learning_rate=param2),
                        alpha, cv=2, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y.ravel())
    mean_accuracy = grid.cv_results_['mean_test_score']
    plt.plot([0.0001, 0.0002, 0.0003, 0.0010], mean_accuracy)
    plt.xlabel('alpha values')
    plt.ylabel('Accuracy')
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def decision_tree_train(X, y, use_grid, plot):
    # Too low max depth will cause under-fitting, too high max depth will cause over-fitting,
    # which will do poor on test data, but do well on training.
    # grid-search can take its time, so we are setting by default previously found optimized hyper-parameters.
    best_criterion = 'entropy'
    best_depth = 13
    best_samp_leaf = 1
    best_samp_split = 5
    # option to use grid-search
    if use_grid:
        print("Using grid-search to find best parameters...")
        param_grid = {'criterion': ['gini', 'entropy'],
                      'max_depth': [5, 13, 20],
                      'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 5]}
        grid = GridSearchCV(DecisionTreeClassifier(random_state=seed), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y)
        best_criterion = grid.best_params_['criterion']
        best_depth = grid.best_params_['max_depth']
        best_samp_split = grid.best_params_['min_samples_split']
        best_samp_leaf = grid.best_params_['min_samples_leaf']
        print("Time elapsed grid-searching: {:.2f}s".format(time.time() - start_time))

    if plot:
        plot_dt_depth(X, y, best_criterion, best_samp_leaf, best_samp_split)

    clf_decision_tree = DecisionTreeClassifier(criterion=best_criterion,
                                               max_depth=best_depth,
                                               min_samples_split=best_samp_split,
                                               min_samples_leaf=best_samp_leaf, random_state=seed)
    print("Doing cross validation 10-fold with best parameters found using GridSearchCV, criterion: %s, max depth: %s, "
          "min samples leaf: %s and min samples split: %s" % (best_criterion, best_depth, best_samp_leaf,
                                                              best_samp_split))

    # Cross validation with 10-folds.
    cross_val_dt_accuracy = cross_val_score(clf_decision_tree, X, y, scoring='accuracy', cv=10)

    # Printing the mean accuracy from all the scores given by the cross validation from each fold.
    print("Decision tree mean accuracy from cross validation with 10 folds:",
          cross_val_dt_accuracy.mean())
    return cross_val_dt_accuracy.mean(), clf_decision_tree, "Decision tree classifier"


def knn_train(X, y, use_grid, plot):
    # grid-search can take its time, so we are setting by default previously found optimized hyper-parameters.
    best_n_neighbors = 5
    best_weight = 'distance'
    best_metric = 'euclidean'
    # option to use grid-search
    if use_grid:
        print("Using grid-search to find best parameters...")
        param_grid = {'n_neighbors': [1, 5, 10, 20],
                      'weights': ['uniform', 'distance'],
                      'metric': ['euclidean', 'manhattan']}
        grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y.ravel())
        best_n_neighbors = grid.best_params_['n_neighbors']
        best_weight = grid.best_params_['weights']
        best_metric = grid.best_params_['metric']
        print("Time elapsed grid-searching: {:.2f}s".format(time.time() - start_time))

    if plot:
        plot_knn_n(X, y, best_weight, best_metric)
    clf_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, metric=best_metric, weights=best_weight)
    print("Doing cross validation 10-fold with best parameters found using GridSearchCV,"
          " n_neighbors: %s, weight: %s, "
          "metric: %s" % (best_n_neighbors, best_weight, best_metric))
    cross_val_knn_accuracy = cross_val_score(clf_knn, X, y.ravel(), scoring='accuracy', cv=10)
    print("Knn mean accuracy from cross validation with 10 folds:", cross_val_knn_accuracy.mean())
    return cross_val_knn_accuracy.mean(), clf_knn, "K nearest neighbor"


def neural_train(X, y, use_grid, plot):
    # grid-search can take its time, so we are setting by default previously found optimized hyper-parameters.
    best_activation = 'relu'
    best_learning_rate = 'constant'
    # can read off from plot the best value of alpha
    best_alpha = 0.0003
    # option to use grid-search
    if use_grid:
        print("Using grid-search to find best parameters...")
        param_grid = {'activation': ['relu', 'tanh', 'logistic'],
                      'learning_rate': ['constant', 'adaptive']}
        grid = GridSearchCV(MLPClassifier(max_iter=500, random_state=seed),
                            param_grid, cv=2, scoring='accuracy', n_jobs=-1)
        grid.fit(X, y.ravel())
        best_activation = grid.best_params_['activation']
        best_learning_rate = grid.best_params_['learning_rate']
        print("Time elapsed grid-searching: {:.2f}s".format(time.time() - start_time))

    if plot:
        plot_nn_alpha(X, y, best_activation, best_learning_rate)

    clf_neural = MLPClassifier(random_state=seed, max_iter=500, activation=best_activation,
                               learning_rate=best_learning_rate, alpha=best_alpha)
    print("Doing cross validation 5-fold with best parameters found using GridSearchCV,"
          " activation: %s, "
          "learning_rate: %s, alpha: %s" % (best_activation, best_learning_rate, best_alpha))
    cross_val_neural_accuracy = cross_val_score(clf_neural, X, y.ravel(), scoring='accuracy', cv=5)
    print("Neural mean accuracy score from cross validation with 5 folds:", cross_val_neural_accuracy.mean())

    return cross_val_neural_accuracy.mean(), clf_neural, "Neural network"


def svc(X, y):
    clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=seed))

    cross_val_svc_accuracy = mean(cross_val_score(clf_svc, X, y.ravel(), scoring='accuracy', cv=3))
    print("SVC mean accuracy score from cross validation with 3 folds:", cross_val_svc_accuracy)

    return cross_val_svc_accuracy.mean(), clf_svc, "Support Vector Classification"


def get_best_model(models):
    best_mean_accuracy = 0
    best_clf = None
    best_name = None

    for model in models:
        if model[0] > best_mean_accuracy:
            best_mean_accuracy = model[0]
            best_clf = model[1]
            best_name = model[2]
    return best_clf, best_name


def predict_best_model(model, X_training, y_training, X_test_set, y_true):
    model.fit(X_training, y_training.ravel())
    y_pred = model.predict(X_test_set)
    score = accuracy_score(y_true, y_pred)
    return score


# Load data, and process if wanted
def load(data_in, label_in, process):
    data_out = pd.read_csv(data_in).to_numpy()
    label_out = pd.read_csv(label_in).to_numpy()
    if process:
        print("Processing data...")
        data_out = one_zero(data_out)
        print("Time elapsed processing: {:.2f}s".format(time.time() - start_time))
    return data_out, label_out


def show_image(X):
    X_image = X[123].reshape(28, 28)
    plt.imshow(X_image, cmap='Greys')
    plt.show()


if __name__ == '__main__':
    seed = 321
    start_time = time.time()

    # Set value True for data preprocessing and False for no preprocessing of the data.
    data_processing = True
    data, data_labels = load("handwritten_digits_images.csv", "handwritten_digits_labels.csv",
                             data_processing)

    # Splitting the data into training and test data. 20% Test data and 80% training data.
    X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=0.2, random_state=seed)
    # show_image(X_train) <- Can show image with preprocessed data and without so you can see the difference.
    # This can be done by uncommenting show_image(X_train) and changing data_processing to True/False.

    # svc(X_train, y_train) <- Add to the models_list if want to include SVC.
    models_list = [decision_tree_train(X_train, y_train, use_grid=False, plot=False),
                   knn_train(X_train, y_train, use_grid=False, plot=False),
                   neural_train(X_train, y_train, use_grid=False, plot=False)]

    best_model = get_best_model(models_list)
    # taking the model with the best cross validation score and using it against the unseen test data
    score_best_model = predict_best_model(best_model[0], X_train, y_train, X_test, y_test)
    print("Accuracy: %s on test set with model: %s" % (score_best_model, best_model[1]))

    print("Time elapsed: {:.2f}s".format(time.time() - start_time))
