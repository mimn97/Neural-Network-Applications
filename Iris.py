'''
// Name: Minhwa Lee
// Assignment: CS200 Junior IS Software
// Title: Software for Iris classification
// Course: CS 200
// Semester: Spring 2020
// Instructor: D. Byrnes
// Date: 04/25/2020
// Sources consulted: 'https://github.com/ashkanmradi/MLP-classifier/blob/master/main.py', 'pandas' and 'scikit-learn' library documentation website
https://pandas.pydata.org/docs/reference/index.html,https://scikit-learn.org/stable/modules/classes.html
// Program description: This program is to execute data classification of iris plants using multilayer perceptron with propagation.
It is written on Python Scikit-learn library.
// Known bugs: I've used 'warnings' to ignore all the unimportant warnings during running the program.
// Creativity: Except for for-loop codes that make a plot, all remaining codes are written by myself.


// Instructions:
1. Before running the program, you should install 'pandas' 'sklearn' and 'matplotlib' libraries first.
2. Then run the program then you will see the accuracy information as well as the plot.
'''

# Modules used for building the MLP model
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == "__main__":

    warnings.filterwarnings(action='ignore') # Ignore warnings

    # Import iris data set directly from the UCI Machine learning website
    iris_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign column names to the data set
    names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
             'Species']

    # Read dataset to pandas dataframe
    data = pd.read_csv(iris_url, names=names)

    # Place all independent variables in X and a dependent variable in Y
    X = data[
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = data[['Species']]

    # Normalize the data set
    norm_data = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Convert text to numeric values for NN performance
    target_col = Y.replace(
        ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2])

    df = pd.concat([norm_data, target_col], axis=1)
    # df is the modified dataset that we are going to apply MLP to.

    ## Data Separation  (Trainset vs Testset)

    X_df = df[
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y_df = df.Species

    # Separate the entire data set into train set and test set, each taking up 70% and 30%.

    train, test = train_test_split(df, test_size=0.3, random_state=1)

    trainX = train[
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    trainY = train.Species

    testX = test[
        ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    testY = test.Species

    # Training and Testing

    clf = MLPClassifier(solver='sgd', alpha=1e-3, max_iter=3000,
                        hidden_layer_sizes=(10,), random_state=1)

    clf.fit(trainX, trainY)

    # Make a prediction and Check the accuracy of the MLP model

    prediction = clf.predict(testX)

    correct_setosa = 0
    correct_versicolor = 0
    correct_virginica = 0

    for i in range(len(prediction)):
        if prediction[i] == 0 and testY.values[i] == 0:
            correct_setosa += 1 # If correctly classify iris-setosa, then increment by 1
        elif prediction[i] == 1 and testY.values[i] == 1:
            correct_versicolor += 1 # If correctly classify iris-versicolor, then increment by 1
        elif prediction[i] == 2 and testY.values[i] == 2:
            correct_virginica += 1 # If correctly classify iris-virginica, then increment by 1

    prediction = list(prediction)
    list_test_y = list(testY.values)

    print("Correctly classified Setosa : ", correct_setosa)
    print("Incorrectly classified Setosa: ", list_test_y.count(0) - correct_setosa)
    print("Correctly classified Versicolor: ", correct_versicolor)
    print("Incorrectly classified Veriscolor: ", list_test_y.count(1) - correct_versicolor)
    print("Correctly classified Virginica: ", correct_virginica)
    print("Incorrectly classified Virginica: ", list_test_y.count(2) - correct_virginica)

    print('The accuracy of the MLP is:',
          accuracy_score(testY, prediction) * 100)

    # Make a plot for mean accuracy score vs number of hidden units in the hidden layer

    lst_accuracy_plt = [] # A list representing accuracy of the MLP with
    # each different number of hidden units

    plt.figure(figsize=(5,5))
    plt.grid(True)
    axe = plt.axes()

    for num_hidden_units in range(1, 21): # Set a maximum hidden units to 20
        clf_hidden = MLPClassifier(solver='sgd', alpha=1e-3, max_iter=3000,
                                   hidden_layer_sizes=(num_hidden_units,),
                                   random_state=1)
        clf_hidden.fit(trainX, trainY) # training process
        clf_prediction = clf_hidden.predict(testX) # Testing process
        clf_hidden_result = accuracy_score(testY, clf_prediction) # Compute Accuracy rate
        lst_accuracy_plt.append(clf_hidden_result*100)
        plt.scatter(num_hidden_units, clf_hidden_result * 100, c='blue')
        axe.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        plt.xlabel('Number of Hidden units in the Hidden layer')
        plt.ylabel('Accuracy scores (%) of MLP ')
        plt.title('Performance of The MLP model for Classifying Iris Species')

    plt.show()
