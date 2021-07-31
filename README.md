# SGD and Random Forest Classifier

#### Content Includes:
- Stochastic Gradient Descent (SGD) Classifier
  - SGD handles very large datasets efficiently. This is in part because SGD deals with training instances independently. The SGDClassifier relies on randomness during training (hence the name “stochastic”).

  `from sklearn.linear_model import SGDClassifier`
  `sgd_clf = SGDClassifier(random_state=42)`
  `sgd_clf.fit(X_train, y_train_5)`
  
- Random Forest Classifier
- Stratified Sampling
- Confusion Matrix

  `from sklearn.model_selection import cross_val_predict`
  `y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)`

![2021-07-31 15_35_39-Classification_Stochastic_Gradient_Descent_(SGD)_Classifier - Jupyter Notebook](https://user-images.githubusercontent.com/44503223/127751895-d0ae4948-8240-48c4-890b-be0cebbb86fd.png)

- ROC and AUC

![ROC and AUC](https://user-images.githubusercontent.com/44503223/127751807-89ab27dc-01ef-4a5c-8f9a-f62b7bb1a40a.png)

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
