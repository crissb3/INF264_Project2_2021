import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time


data = pd.read_csv("data/handwritten_digits_images.csv").to_numpy()
data_labels = pd.read_csv("data/handwritten_digits_labels.csv").to_numpy()


def decision_tree(dataset, dataset_labels):
    # Too low max depth will cause underfitting, too high max depth will cause overfitting,
    # which will do poor on test data, but do well on training.
    #max_depth = [5, 10, 13, 15, 17, 20, 40]
    #max_depth = [15, 16, 17, 40]
    max_depth = [15, 17] # best = 15 og 17
    min_samples_split_a = [2,3,4,5] # best = 2
    #min_samples_leaf_a = [1,2,5] # Best = 5
    #for n in min_samples_leaf_a:
    #print("MIN SAMPLES LEAF: ", n)
    clf_decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=17, min_samples_split=2,
                                               min_samples_leaf=5)
    X_train, X_test, y_train, y_test = train_test(dataset, dataset_labels)
    #X_train, X_val, y_train, y_val = train_val_split(X_train, y_train)

    # Doing Kfold cross validation to see if the average score of 5 or 10 folds gives a better score/accuracy
    # than if we trained the data on 60% of the set and then did a predict on validation data (eg. 20%).
    # Doing cross validation we split the set into 5 or 10 equally sized smaller datasets where the split
    # is either 4 training sets and 1 validation set or 9 training sets and 1 validation set.
    # The validation set shifts for each fold. For example: [V, T,T, T, T] for the first fold,
    # [T, V, T, T, T] for the second fold and so on, where T = Training and V = Validation.
    scores_kfold_decision_tree_RMSE = cross_val_score(clf_decision_tree, X_train, y_train,
                                                 scoring='neg_root_mean_squared_error', cv=10)
    scores_kfold_decision_tree_accuracy = cross_val_score(clf_decision_tree, X_train, y_train, scoring='accuracy',
                                                          cv=10)

    # Prints the mean score of all folds, giving an average score of all the scores from each fold. After some
    # testing back and forth changing up number of folds we found that 10 gives a slightly better score than 5,
    # and about 0.5% to 1% (depending on model) better accuracy than if we were to train our models without cross
    # validation.
    print("Decision tree mean RMSE score from kfold with 10 folds:", abs(mean(scores_kfold_decision_tree_RMSE)))
    # Printing the mean accuracy from all folds with standard deviation.
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_kfold_decision_tree_accuracy.mean(),
          scores_kfold_decision_tree_accuracy.std()*2))

    #clf_decision_tree.fit(X_train, y_train)
    #y_pred = clf_decision_tree.predict(X_val)
    #rmse_knn = mean_squared_error(y_val, y_pred, squared=False)
    #print("Decision tree RMSE after prediction on validation set:", rmse_knn)
    #print("Accuracy score for decision tree after prediction on validation data:", accuracy_score(y_val, y_pred))


    # Reshaping validation dataset to 28x28 to show image.
    #reshaped_X_val = np.array(X_val).reshape([X_val.shape[0], 28, 28])
    #d = reshaped_X_val[121]
    #plt.imshow(d, cmap='Greys')
    #plt.show()
    #print(clf.predict([X_val[121]]))


def knn(dataset, dataset_labels):
    #n = [1, 2, 5, 10, 15]
    #best = 5
    #best_n = 1
    #for i in n:
    #print("RUNNING KNN WITH N = " + str(i))
    clf_knn = KNeighborsClassifier(n_neighbors=1)
    X_train, X_test, y_train, y_test = train_test(dataset, dataset_labels)
    #X_train, X_val, y_train, y_val = train_val_split(X_train, y_train)

    scores_kfold_knn = cross_val_score(clf_knn, X_train, y_train.ravel(), scoring='neg_root_mean_squared_error', cv=10)
    print("Knn RMSE score from mean of all kfold scores:", abs(mean(scores_kfold_knn)))
    scores_kfold_knn_accuracy = cross_val_score(clf_knn, X_train, y_train.ravel(), scoring='accuracy',
                                                          cv=10)
    print("Knn mean accuracy from cross validation 10-fold :", scores_kfold_knn_accuracy.mean())

    #clf_knn.fit(X_train, y_train.ravel())
    #y_pred = clf_knn.predict(X_val)
    #rmse_knn = mean_squared_error(y_val, y_pred, squared=False)
    #print("Knn RMSE after prediction on validation set:", rmse_knn)
    #    if rmse_knn < best:
    #        best = rmse_knn
    #        best_n = i
    #print("Best neighbor:", best_n)
    #print("Best rmse =", best)


def svc(dataset, dataset_labels):
    clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    X_train, X_test, y_train, y_test = train_test(dataset, dataset_labels)
    scores_kfold_svc = cross_val_score(clf_svc, X_train, y_train.ravel(), scoring='neg_root_mean_squared_error', cv=10)
    print("SVC RMSE score from mean of all kfold scores:", abs(mean(scores_kfold_svc)))
    scores_kfold_svc_accuracy = cross_val_score(clf_svc, X_train, y_train.ravel(), scoring='accuracy',
                                                cv=10)
    print("SVC mean accuracy from cross validation 10-fold :", scores_kfold_svc_accuracy.mean())

def train_test(dataset, dataset_labels):
    # Splitting the data into training and test data. 20% Test data and 80% training data.
    X_train, X_test, y_train, y_test = train_test_split(dataset, dataset_labels, test_size=0.2, random_state=14)
    return X_train, X_test, y_train, y_test


def train_val_split(X_train, y_train):
    # Splitting the train data into training and validation data. Since 80% of the data is now training data, we split
    # such that 0.25 of the training data becomes validation data. 0.25 * 0.8 = 0.2 = 20% validation data.
    # This gives a 60% Training, 20% Validation and 20% Test data split.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=99)
    return X_train, X_val, y_train, y_val


def get_model_score(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)


if __name__ == '__main__':
    start_time = time.time()
    #!!!decision_tree(data, data_labels)
    #!!!knn(data, data_labels)
    svc(data, data_labels)
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
