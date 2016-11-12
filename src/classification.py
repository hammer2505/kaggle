import time
from sklearn import metrics
from sklearn import cross_validation


class classifier:

    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(self):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
        return model

    # KNN Classifier
    def knn_classifier(self):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        return model

    # Logistic Regression Classifier
    def logistic_regression_classifier(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        return model

    # Random Forest Classifier
    def random_forest_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=16)
        return model

    # Decision Tree Classifier
    def decision_tree_classifier(self):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        return model

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        return model

    # SVM Classifier
    def svm_classifier(self):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        return model

# end of class classifier define

m_classifier = classifier()
# m_classifiers_name = ['KNN', 'LR', 'RF', 'DT', 'GBDT']
m_classifiers_name = ['LR']
m_classifiers = {'KNN': m_classifier.knn_classifier(),
                 'LR': m_classifier.logistic_regression_classifier(),
                 'RF': m_classifier.random_forest_classifier(),
                 'DT': m_classifier.decision_tree_classifier(),
                 'GBDT': m_classifier.gradient_boosting_classifier()
                 }


def TraininAllClassifiers(
        classifiers, train_data, train_label, test_data, test_label):
    num_train = train_data.shape[0]
    num_feat = train_data.shape[1]
    num_test = test_data.shape[0]
    num_feat = test_data.shape[1]
    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (
        num_train, num_test, num_feat)

    for classifiers_name_it in m_classifiers_name:
        print '****************** %s ******************' % classifiers_name_it
        start_time = time.time()
        classifiers[classifiers_name_it] = classifiers[classifiers_name_it].fit(train_data, train_label)
        print 'training took %fs!' % (time.time() - start_time)
        predict = classifiers[classifiers_name_it].predict(test_data)
        accuracy = metrics.accuracy_score(test_label, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)


def save_classifiers(classifiers, filename="../classifiers_model"):
    print("******************** save classifier model into file: " + filename)
    import pickle
    pickle.dump(classifiers, open(filename, 'wb'))


def load_classifiers(filename="../classifiers_model"):
    print("******************** load classifier model from file: " + filename)
    import pickle
    classifiers = pickle.load(open(filename, 'rb'))
    return classifiers


def PrediectinAllClassifiers(classifiers, test_data):
    num_test = test_data.shape[0]
    num_feat = test_data.shape[1]
    print '******************** Data Info *********************'
    print '#testing_data: %d, dimension: %d' % (num_test, num_feat)
    predictRes = {
        'KNN': [],
        'LR': [],
        'RF': [],
        'DT': [],
        'GBDT': []
        }
    for classifiers_name_it in m_classifiers_name:
        print "%s is predicting" % (classifiers_name_it)
        predictRes[classifiers_name_it] = \
            classifiers[classifiers_name_it].predict(test_data)
    return predictRes


def CrossValidateClassifiers(
        classifiers, times, num_fold, train_data, train_label):
    CrossValidationScore = {'KNN': 0,
                            'LR': 0,
                            'RF': 0,
                            'DT': 0,
                            'GBDT': 0
                            }
    for classifiers_name_it in m_classifiers_name:
        for times_it in range(times):
            CrossValidationScore[classifiers_name_it] = \
                CrossValidationScore[classifiers_name_it] + \
                sum(cross_validation.cross_val_score(
                    classifiers[classifiers_name_it], train_data, train_label, cv=num_fold)) / num_fold
    for classifiers_name_it in m_classifiers_name:
        CrossValidationScore[classifiers_name_it] = \
            CrossValidationScore[classifiers_name_it] / times
    return CrossValidationScore


# get the mode of different classifiers, present a vote function
def ModethePredict(test_data, predictRes):
    from scipy.stats import mode
    predictMode = []
    for i in range(len(test_data)):
        predictMode.append(
            mode([predictRes['LR'][i]])[0][0])
    return predictMode
