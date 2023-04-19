from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def get_metrics(preds, labels):
    """Takes predictions and labels and generates accuracy,
    precision and recall scores and retuerns them in that order.
    """
    
    acc = accuracy_score(labels, preds)
    p = precision_score(labels, preds) # tp / (tp + fp)
    r = recall_score(labels, preds) # tp / (tp + fn)
    
    return acc, p, r


def classify_logistic_reg(
    train_features, 
    train_labels, 
    test_features, 
    test_labels
):
    """Training a logistric regression model and calculating 
    test and train scores.

    Returns:
        Train and Test Metrics
    """
    
    # Training (.fit) on train_features and labels. Using .predict to
    # get predictions (0 or 1).
    clf = LogisticRegression(random_state=0).fit(train_features, train_labels)
    train_preds = clf.predict(train_features)
    train_metrics = get_metrics(train_preds, train_labels)
    test_preds = clf.predict(test_features)
    test_metrics = get_metrics(test_preds, test_labels)
    
    return train_metrics, test_metrics


def classify_decision_tree(
    train_features, 
    train_labels, 
    test_features, 
    test_labels
):
    """Training a decision tree model and calculating 
    test and train scores.

    Returns:
        Train and Test Metrics
    """
    
    # Training (.fit) on train_features and labels. Using .predict to
    # get predictions (0 or 1).
    clf = DecisionTreeClassifier(random_state=0).fit(train_features, train_labels)
    train_preds = clf.predict(train_features)
    train_metrics = get_metrics(train_preds, train_labels)
    test_preds = clf.predict(test_features)
    test_metrics = get_metrics(test_preds, test_labels)
    
    return train_metrics, test_metrics


def classify_random_forest(
    train_features, 
    train_labels, 
    test_features, 
    test_labels
):
    """Training a Random Forest model and calculating 
    test and train scores.

    Returns:
        Train and Test Metrics
    """
    
    # Training (.fit) on train_features and labels. Using .predict to
    # get predictions (0 or 1).
    clf = RandomForestClassifier(random_state=0).fit(train_features, train_labels)
    train_preds = clf.predict(train_features)
    train_metrics = get_metrics(train_preds, train_labels)
    test_preds = clf.predict(test_features)
    test_metrics = get_metrics(test_preds, test_labels)
    
    return train_metrics, test_metrics


def classify_svm(
    train_features, 
    train_labels, 
    test_features, 
    test_labels
):
    """Training a Support Vector Machine model and calculating 
    test and train scores.

    Returns:
        Train and Test Metrics
    """
    
    # Training (.fit) on train_features and labels. Using .predict to
    # get predictions (0 or 1).
    clf = svm.SVC(random_state=0).fit(train_features, train_labels)
    train_preds = clf.predict(train_features)
    train_metrics = get_metrics(train_preds, train_labels)
    test_preds = clf.predict(test_features)
    test_metrics = get_metrics(test_preds, test_labels)
    
    return train_metrics, test_metrics


def classify_knn(
    train_features, 
    train_labels, 
    test_features, 
    test_labels
):
    """Training a K-nearest Neighbors Machine model and calculating 
    test and train scores.

    Returns:
        Train and Test Metrics
    """
    
    # Training (.fit) on train_features and labels. Using .predict to
    # get predictions (0 or 1).
    clf = KNeighborsClassifier(n_neighbors=3).fit(train_features, train_labels)
    train_preds = clf.predict(train_features)
    train_metrics = get_metrics(train_preds, train_labels)
    test_preds = clf.predict(test_features)
    test_metrics = get_metrics(test_preds, test_labels)
    
    return train_metrics, test_metrics
