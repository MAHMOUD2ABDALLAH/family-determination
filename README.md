# Family Members Segmentation Report

This repository contains the code and documentation for a classification project. The project involves preprocessing and analyzing a dataset with categorical features to prepare it for classification tasks. Various classification models are applied to evaluate their performance, and decision boundaries are visualized to understand the models' behavior.

---

## Project Overview

The goal of this project is to preprocess and analyze a dataset containing categorical features for classification purposes. The dataset includes features such as `Gender`, `Ever Married`, `Graduated`, `Profession`, `Spending_Score`, and `Var_1`. The preprocessing steps involve encoding categorical variables, handling missing values, scaling numerical features, and selecting the most impactful features for prediction. Multiple classification models are then applied to evaluate their performance, and decision boundaries are visualized to understand the models' behavior.

---

## Repository Structure

- **/docs**: Contains project documentation, including preprocessing steps and analysis results.
- **/data**: Includes the dataset used for the project.
- **/scripts**: Contains the preprocessing and classification scripts.
- **/visualizations**: Contains scripts and images for decision boundary visualizations.
- **README.md**: This file, providing an overview of the project.

---

## Preprocessing Techniques

### 1. Encoding Categorical Features
- **Categorical Features**: `Gender`, `Ever Married`, `Graduated`, `Profession`, `Spending_Score`, `Var_1`
- **Encoding**: Converted categorical features to numerical values using `LabelEncoder`.
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoding = LabelEncoder()
  seg['Gender'] = encoding.fit_transform(seg['Gender'])
  seg['Ever Married'] = encoding.fit_transform(seg['Ever Married'])
  seg['Graduated'] = encoding.fit_transform(seg['Graduated'])
  seg['Profession'] = encoding.fit_transform(seg['Profession'])
  seg['Spending_Score'] = encoding.fit_transform(seg['Spending_Score'])
  seg['Var_1'] = encoding.fit_transform(seg['Var_1'])
  ```

### 2. Handling Missing Values
- **Imputation**: Used `SimpleImputer` to replace missing values with the most frequent value in each column.
  ```python
  from sklearn.impute import SimpleImputer
  imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
  seg['Ever Married'] = imp.fit_transform(seg['Ever Married'].reshape(-1, 1))
  seg['Graduated'] = imp.fit_transform(seg['Graduated'].reshape(-1, 1))
  seg['Profession'] = imp.fit_transform(seg['Profession'].reshape(-1, 1))
  seg['Work Experience'] = imp.fit_transform(seg['Work Experience'].reshape(-1, 1))
  seg['Family Size'] = imp.fit_transform(seg['Family Size'].reshape(-1, 1))
  seg['Var_1'] = imp.fit_transform(seg['Var_1'].reshape(-1, 1))
  ```

### 3. Scaling Numerical Features
- **Scaling**: Applied `MinMaxScaler` to scale numerical features to a range of (0, 1).
  ```python
  from sklearn.preprocessing import MinMaxScaler
  scale = MinMaxScaler(copy=True, feature_range=(0, 1))
  seg['Age'] = scale.fit_transform(seg['Age'].reshape(-1, 1))
  seg['Family Size'] = scale.fit_transform(seg['Family Size'].reshape(-1, 1))
  ```

### 4. Feature Selection
- **Feature Selection**: Used `SelectFromModel` with `RandomForestRegressor` to select the most impactful features.
  ```python
  from sklearn.feature_selection import SelectFromModel
  from sklearn.ensemble import RandomForestRegressor
  select2 = SelectFromModel(RandomForestRegressor())
  selected = select2.fit_transform(x, y)
  print(selected.shape)
  print(select2.get_support())
  ```
- **Output**: The selected features are indicated by `True` values. However, due to overfitting concerns, all features were used for the final model.

---

## Classification Models

Various classification models were applied to evaluate their performance:

### Ensemble Models
- **Gradient Boosting Classifier**:
  ```python
  from sklearn.ensemble import GradientBoostingClassifier
  model1 = GradientBoostingClassifier(learning_rate=0.04)
  model1.fit(x_train, y_train)
  y_pred = model1.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **Random Forest Classifier**:
  ```python
  from sklearn.ensemble import RandomForestClassifier
  model2 = RandomForestClassifier()
  model2.fit(x_train, y_train)
  y_pred = model2.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

### Normal Models
- **Logistic Regression**:
  ```python
  from sklearn.linear_model import LogisticRegression
  model3 = LogisticRegression()
  model3.fit(x_train, y_train)
  y_pred = model3.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **K-Nearest Neighbors (KNN)**:
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  model5 = KNeighborsClassifier()
  model5.fit(x_train, y_train)
  y_pred = model5.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **Decision Tree**:
  ```python
  from sklearn.tree import DecisionTreeClassifier
  model6 = DecisionTreeClassifier(max_depth=10)
  model6.fit(x_train, y_train)
  y_pred = model6.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **Naive Bayes**:
  ```python
  from sklearn.naive_bayes import GaussianNB, BernoulliNB
  model7 = GaussianNB()
  model8 = BernoulliNB()
  model7.fit(x_train, y_train)
  y_pred = model7.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **Linear Discriminant Analysis (LDA)**:
  ```python
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  model9 = LinearDiscriminantAnalysis()
  model9.fit(x_train, y_train)
  y_pred = model9.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

- **Quadratic Discriminant Analysis (QDA)**:
  ```python
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  model10 = QuadraticDiscriminantAnalysis()
  model10.fit(x_train, y_train)
  y_pred = model10.predict(x_train)
  print('Accuracy score:', accuracy_score(y_train, y_pred))
  ```

---

## Model Performance

The accuracy scores for the different models are as follows:

1. **Logistic Regression**: 0.2559
2. **K-Nearest Neighbors (KNN)**: 0.5436
3. **Decision Tree Classifier (max_depth=10)**: 0.6291
4. **Linear Discriminant Analysis (LDA)**: 0.4524
5. **Quadratic Discriminant Analysis (QDA)**: 0.2503
6. **Gaussian Naive Bayes**: 0.2559
7. **Bernoulli Naive Bayes**: 0.4202

Ensemble models, such as Random Forest and Gradient Boosting, were found to be the most effective for both regression and classification tasks.

---

## Voting Classifier

A Voting Classifier was used to combine the predictions of multiple models:

```python
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(
    estimators=[('1', model1), ('2', model2), ('3', model3), ('5', model5), ('6', model6),
               ('7', model7), ('8', model8), ('9', model9), ('10', model10)], voting='hard')
eclf.fit(x_train, y_train)
y_pred = eclf.predict(x_train)
print('Accuracy score:', accuracy_score(y_train, y_pred))
```

---

## Decision Boundary Visualization

The decision boundaries for the models were visualized to understand their performance:

```python
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

X, Y = make_classification(n_samples=7165, n_features=2, n_informative=2, n_redundant=0, n_classes=2)

gs = gridspec.GridSpec(4, 2)
fig = plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

labels = ['GradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'KNN',
          'DecisionTreeClassifier', 'GaussianNB', 'BernoulliNB', 'LinearDiscriminantAnalysis',
          'QuadraticDiscriminantAnalysis']

for clf, lab, grd in zip([model1, model2, model3, model5, model6, model7, model8, model9, model10],
                          labels, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]):
    clf.fit(X, Y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=Y, clf=clf, legend=2)
    plt.title(lab)

plt.show()
```
![Screenshot 2025-02-15 220656](https://github.com/user-attachments/assets/71ff053f-9005-40da-ae81-588c94901ab2)


---

## Cross-Validation

Cross-validation was performed to evaluate the models' performance:

```python
from sklearn.model_selection import cross_val_score

for clf, label in zip([model1, model2, model3, model5, model6, model7, model8, model9, model10, eclf],
                      ['GradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'KNN',
                       'DecisionTreeClassifier', 'GaussianNB', 'BernoulliNB', 'LinearDiscriminantAnalysis',
                       'QuadraticDiscriminantAnalysis']):
    scores = cross_val_score(clf, X, Y, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
```
![Screenshot 2025-02-15 220638](https://github.com/user-attachments/assets/951c85d4-55ad-4732-ab21-fc764e07e964)

## Cross-Validation Accuracy Scores

- **GradientBoostingClassifier**: 0.48 (+/- 0.01)
- **RandomForestClassifier**: 0.44 (+/- 0.01)
- **LogisticRegression**: 0.25 (+/- 0.01)
- **KNN**: 0.31 (+/- 0.01)
- **DecisionTreeClassifier**: 0.43 (+/- 0.01)
- **GaussianNB**: 0.25 (+/- 0.00)
- **BernoulliNB**: 0.42 (+/- 0.01)
- **LinearDiscriminantAnalysis**: 0.45 (+/- 0.01)
- **QuadraticDiscriminantAnalysis**: 0.25 (+/- 0.01)

## Getting Started

### Prerequisites

Before getting started, ensure you have the following:

1. **Python**: Make sure Python is installed on your system.
2. **Libraries**: Install the required libraries using pip:
   ```bash
   pip install numpy pandas scikit-learn matplotlib mlxtend
   ```

```

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/classification-report.git
   cd classification-report
   ```

2. **Run the Scripts**:
   - Navigate to the `/scripts` directory.
   - Execute the preprocessing script to prepare the data:
     ```bash
     python preprocessing.py
     ```

---

## Usage

1. **Preprocess the Data**: Run the preprocessing script to encode categorical features, handle missing values, scale numerical features, and select impactful features.
2. **Analyze the Data**: Use the preprocessed data for classification tasks.
3. **Evaluate Models**: Apply various classification models and evaluate their performance.
4. **Visualize Decision Boundaries**: Use the provided scripts to visualize decision boundaries for different models.

---

## Contributing

We welcome contributions to the **Classification Report** project! If you'd like to contribute, please follow these steps:

1. **Fork the Repository**: Create a fork of the repository on your GitHub account.
2. **Create a Branch**: Make a new branch for your feature or bug fix.
3. **Make Changes**: Implement your changes and ensure they are well-documented.
4. **Submit a Pull Request**: Submit a pull request with a detailed description of your changes.

Please ensure your code follows the project's coding standards and includes appropriate documentation.

---

## Acknowledgments

- **Team Members**: A special thanks to all contributors and team members who worked on this project.
- **Open Source Community**: We are grateful for the tools, libraries, and resources provided by the open-source community.

---

## Contact

For questions, feedback, or collaboration opportunities, please contact:

- **Your Name**: Mahmoud_abdallah20@outlook.com.
- **GitHub Issues**: Open an issue in the repository for technical inquiries.

---

Thank you for your interest in the **Family Members Segmentation Report** project. We hope this system can make a meaningful impact in data preprocessing and classification tasks.

<img width="100" height="100" alt="Upwork" src="https://github.com/user-attachments/assets/3bee725e-f9a7-47a5-83c9-321516bb12c9" />

