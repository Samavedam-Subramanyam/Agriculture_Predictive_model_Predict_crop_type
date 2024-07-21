# Crop Type Prediction Project

## Project Description
This project involves the application of machine learning to agricultural data. The goal is to predict the optimal crop type for a given field based on soil measurements. The model identifies the single feature that has the strongest predictive performance for classifying crop types.

## Dataset
The dataset used in this project is `soil_measures.csv`, which contains the following features:

- "N": Nitrogen content ratio in the soil
- "P": Phosphorous content ratio in the soil
- "K": Potassium content ratio in the soil
- "pH": pH value of the soil
- "crop": Categorical values that contain various crops (target variable)

Each row in this dataset represents various measures of the soil in a particular field. The crop specified in the "crop" column is the optimal choice for that field.

## Methodology
The project involves the following steps:

1. **Data Preparation**: The dataset is split into features (X) and target variable (y). The data is then split into training and testing sets.

2. **Model Building & Evaluation**: A logistic regression model is trained for each feature. The model's performance is evaluated using the F1-score.

3. **Feature Importance**: The feature that produces the best F1-score is identified as the best predictive feature.

## Results
The feature 'K' (Potassium content ratio in the soil) was identified as the best predictive feature with an F1-score of 0.2584. This suggests that the potassium content in the soil has the strongest predictive performance for classifying crop types in this dataset.

## Conclusion
This project demonstrates the application of machine learning in agriculture, particularly in helping farmers make data-driven decisions on crop selection based on soil condition. The model can be further improved by incorporating more features and using more complex algorithms.

## Code
The Python code used for this project is provided below:

```python
# Importing required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Splitting the data
X = crops[["N", "P", "K", "ph"]]
y = crops["crop"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

# Evaluating performance
dic = {}
for feature in ["N", "P", "K", "ph"]:
    logreg = LogisticRegression()
    logreg.fit(X_train[[feature]], y_train)
    y_pred = logreg.predict(X_test[[feature]])
    feature_performance = metrics.f1_score(y_test, y_pred, average='weighted')
    dic[feature] = feature_performance

# Finding the best predictive feature
best_predictive_feature = {max(dic, key=dic.get): dic[max(dic, key=dic.get)]}
```
