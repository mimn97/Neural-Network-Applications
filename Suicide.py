'''
// Name: Minhwa Lee
// Assignment: CS200 Junior IS Software
// Title: Software for Predicting High Risk of Suicide among Adolescents in South Korea
// Course: CS 200
// Semester: Spring 2020
// Instructor: D. Byrnes
// Date: 04/25/2020
// Sources consulted: 'https://github.com/ashkanmradi/MLP-classifier/blob/master/main.py', 'pandas' and 'scikit-learn' library documentation website
https://pandas.pydata.org/docs/reference/index.html,https://scikit-learn.org/stable/modules/classes.html
// Program description: This program is to execute data classification/prediction of high risk of suicide using multilayer perceptron with propagation.
It is written on Python Scikit-learn library.
// Known bugs: I've used 'warnings' to ignore all the unimportant warnings during running the program.
// Creativity: Except for for-loop codes that make a plot, all remaining codes are written by myself.


// Instructions:
1. Before running the program, you should install 'pandas' 'sklearn' and 'matplotlib' libraries first.
2. Then run the program then you will see the accuracy information as well as the plot.
'''


import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":

    warnings.filterwarnings(action='ignore') # Ignore warnings

    # Import csv suicide data set from local directory
    os.listdir(os.getcwd())
    data = pd.read_csv("kyrbs2019_edit.csv", decimal=',')

    data = pd.DataFrame(data) # Make an imported csv file a Dataframe that are usable in pandas library

    # Generate a sample with only 5% of its original data size.
    data = data.sample(frac=0.05, random_state=1)

    ## Data Cleaning

    # Reconstruct a data set with only selected features (independent variables vs dependent variable)
    dat = data [['SEX', 'E_EDU_F', 'E_EDU_F', 'E_S_RCRD', 'E_SES', 'TC_LT',
                  'AC_LT', 'DR_EXP', 'PA_VIG', 'S_SI', 'M_SAD', 'M_STR',
                  'PR_BI', 'PR_HT', 'M_SLP_EN', 'I_SCH_TRT', 'V_TRT', 'M_SUI_CON']].astype('int64')

    # Place all independent variables in X and a dependent variable in Y
    X = dat [['SEX', 'E_EDU_F', 'E_EDU_F', 'E_S_RCRD', 'E_SES', 'TC_LT',
                  'AC_LT', 'DR_EXP', 'PA_VIG', 'S_SI', 'M_SAD', 'M_STR',
                  'PR_BI', 'PR_HT', 'M_SLP_EN', 'I_SCH_TRT', 'V_TRT']]

    # A Dependent variable
    Y = dat[['M_SUI_CON']] # Dependent Variable - Suicide Attempt

    # Data Normalization for independent variables set X
    norm_data = X.apply(lambda x: (x-x.min()) / (x.max() - x.min()))

    # df is the modified dataset that we are going to apply MLP to.
    df = pd.concat([norm_data, Y], axis=1)

    # Data Separation

    # Separate the entire data set into train set and test set, each taking up 70% and 30%.
    train, test = train_test_split(df, test_size=0.3, random_state=1)

    trainX = train[['SEX', 'E_EDU_F', 'E_EDU_F', 'E_S_RCRD', 'E_SES', 'TC_LT',
                  'AC_LT', 'DR_EXP', 'PA_VIG', 'S_SI', 'M_SAD', 'M_STR',
                  'PR_BI', 'PR_HT', 'M_SLP_EN', 'I_SCH_TRT', 'V_TRT']]

    trainY = train[['M_SUI_CON']]

    testX = test[['SEX', 'E_EDU_F', 'E_EDU_F', 'E_S_RCRD', 'E_SES', 'TC_LT',
                  'AC_LT', 'DR_EXP', 'PA_VIG', 'S_SI', 'M_SAD', 'M_STR',
                  'PR_BI', 'PR_HT', 'M_SLP_EN', 'I_SCH_TRT', 'V_TRT']]

    testY = test[['M_SUI_CON']]

    # Training and Testing

    clf = MLPClassifier(solver='sgd', alpha=1e-3, max_iter=3000,
                        hidden_layer_sizes=(10,), random_state=1)

    clf.fit(trainX, trainY)

    # Make a prediction and Check the accuracy of the MLP model
    prediction = clf.predict(testX)

    correct_1 = 0
    correct_2 = 0
    for i in range(len(prediction)):
        if prediction[i] == 1 and testY.values[i] == 1:
            correct_1 += 1 # If correctly classify 'no' then increment by 1
        elif prediction[i] == 2 and testY.values[i] == 2:
            correct_2 += 1 # if correctly classify 'yes' then increment by 1

    prediction = list(prediction)
    total_test_y = list(testY.values)

    print("Correctly classified No: ", correct_1)
    print("Incorrectly classified No: ", total_test_y.count(1) - correct_1)
    print("Correctly classified Yes: ", correct_2)
    print("Incorrectly classified Yes: ", total_test_y.count(2) - correct_2)

    print('The accuracy of the MLP is:', accuracy_score(testY, prediction) * 100)

    # Make a plot for mean accuracy score vs number of hidden units in the hidden layer

    lst_accuracy = []

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
        lst_accuracy.append(clf_hidden_result*100)
        plt.scatter(num_hidden_units, clf_hidden_result * 100, c='red')
        axe.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        plt.xlabel('Number of Hidden units in the Hidden layer')
        plt.ylabel('Accuracy scores (%) of MLP ')
        plt.title('Performance of The MLP model on Suicide Data set')

    print(lst_accuracy)
    plt.show()
