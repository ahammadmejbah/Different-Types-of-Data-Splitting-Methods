<div align="center">
      <h1> <img src="#" width="80px"><br/>Different types of data splitting methods</h1>
     </div>

In order to prevent overfitting and guarantee that our model can generalize to new data, data splitting is essential in machine learning. Let's examine a few typical data splitting techniques:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5658374%2F1104d9898cd56b7c9c7d70141c23a3fd%2F1_train-test-split_0.jpg?generation=1697623539235388&alt=media)

Image Credit: Train test split procedure. | Image: Michael Galarnyk | Built In




1. **Train/Test Split**
    This is the simplest method. We split our data into a training set and a testing set.

    ```python
    from sklearn.model_selection import train_test_split

    X, y = [...]  # Your data and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    ```

2. **K-Fold Cross Validation**
    This method involves splitting the data into 'k' subsets. The model is trained on k-1 of these folds and tested on the remaining one. This process is repeated k times, each time with a different fold as the test set.

    ```python
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train and test your model here
    ```

3. **Stratified K-Fold Cross Validation**
    Like K-Fold, but it ensures that each fold maintains the same distribution of classes as the entire dataset.

    ```python
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train and test your model here
    ```

4. **Time Series Split**
    Useful for time series data. In each split, the test set consists of the next 'n' points in the data. This avoids "looking into the future" during training.

    ```python
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train and test your model here
    ```
5. **Leave One Out Cross Validation (LOOCV)**
    This involves training on all data points except one and testing on that single left out point. This is repeated for all data points. It's computationally intensive but can be useful for small datasets.

    ```python
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train and test your model here
    ```
 6. **Stratified Sampling**

 Stratified sampling ensures that the training and test sets have approximately the same percentage of samples of each target class as the complete set.

 ```python
 from sklearn.model_selection import train_test_split

 # Assume X is your feature matrix and y is your labels
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 ```

 7. **Group K-Fold Cross-Validation**

 Group K-Fold cross-validation is a variation of k-fold cross-validation that ensures the same group is not represented in both the training and test sets.

 ```python
 from sklearn.model_selection import GroupKFold

 groups = [...]  # This needs to be a list of group identifiers corresponding to each observation in X
 gkf = GroupKFold(n_splits=5)
 for train_index, test_index in gkf.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Training and testing model
 ```

**In the above examples, `X` and `y` are your feature matrix and label vector respectively. Also, you need to have sklearn installed in your environment.**


Always be sure that no information from the test set leaks into the training set with your data. This is crucial in instances like time series forecasting, where utilizing past data to predict the future may provide erroneous findings. 

## Credit <a href ="https://github.com/BytesOfIntelligences">BytesOfIntelligence</a>

    
