import PreprocessData as pp
import TestRun

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib as plt

# This file is used to test other machine learning algorithms

if __name__ == '__main__':
    pass
    data_trains, data_tests = pp.preprocessing_cross_valid(2012, 2014, 9)
    print("Tests")
    errs = []
    for i in range(9):

        x_train = data_trains[i][:, 1:]
        y_train = data_trains[i][:, 0]

        x_test = data_tests[i][:, 1:]
        y_test = data_tests[i][:, 0]

        # logistic regression
        reg = LogisticRegression()
        reg.fit(x_train, y_train)
        print("Error:", reg.score(x_test, y_test))

        # support vector machine
        # sv = LinearSVC()
        # sv.fit(x_train, y_train)
        # print("Error:", sv.score(x_test, y_test))

        # random forest
        # rf = RandomForestClassifier(n_estimators=100)
        # rf.fit(x_train, y_train)
        # print("Error:", rf.score(x_test, y_test))

        # extremely random trees
        # et = ExtraTreesClassifier(n_estimators=100)
        # et.fit(x_train, y_train)
        # print("Error:", et.score(x_test, y_test))

        pass

