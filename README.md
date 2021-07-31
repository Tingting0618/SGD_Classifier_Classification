# SGD and Random Forest Classifier

#### Content Includes:
- Stochastic Gradient Descent (SGD) Classifier
  - SGD handles very large datasets efficiently. This is in part because SGD deals with training instances independently. The SGDClassifier relies on randomness during training (hence the name “stochastic”).

  ```python
  from sklearn.linear_model import SGDClassifier
  sgd_clf = SGDClassifier(random_state=42)
  sgd_clf.fit(X_train, y_train_5)
  sgd_clf.predict([some_digit])
  ```

- Random Forest Classifier

  ```python
  from sklearn.ensemble import RandomForestClassifier
  forest_clf = RandomForestClassifier(random_state=42)
  forest_clf.fit(X_train, y_train_5)
  forest_clf.predict([some_digit])
  ```

- Confusion Matrix

  **SGD**
  ```python
  from sklearn.model_selection import cross_val_predict
  y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
  from sklearn.metrics import confusion_matrix
  confusion_matrix(y_train_5, y_train_pred)
  ```
  **Random Forest**
  ```python
  from sklearn.model_selection import cross_val_predict
  y_train_pred_rf = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
  from sklearn.metrics import confusion_matrix
  confusion_matrix(y_train_5, y_train_pred_rf)
  ```  
  
  Note: cross_val_predict() performs K-fold cross-validation, but instead of returning the evaluation scores, it returns the predictions made on each test fold. This means that we get a clean prediction for each instance in the training set (“clean” meaning that the prediction is made by a model that never saw the data during training).

- ROC and AUC

  A high-precision classifier is not very useful if its recall is too low!

  ```python

  from sklearn.metrics import roc_curve

  y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
  fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

  y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,method="predict_proba")
  y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
  fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

  from sklearn.metrics import roc_auc_score
  roc_auc_score(y_train_5, y_scores)
  roc_auc_score(y_train_5, y_scores_forest)
  ```

![ROC and AUC](https://user-images.githubusercontent.com/44503223/127751807-89ab27dc-01ef-4a5c-8f9a-f62b7bb1a40a.png)

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
