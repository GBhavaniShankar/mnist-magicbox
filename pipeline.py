from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from joblib import dump
import pandas as pd
import numpy as np

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

df = pd.DataFrame(X)
df['target'] = y
df_sampled = df.sample(n=20000, random_state=42)

X_sampled = df_sampled.drop(columns='target').to_numpy()
y_sampled = df_sampled['target'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.2, stratify=y_sampled, random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42, whiten=True)),
    ('clf', OneVsRestClassifier(SVC()))
])

search_space = {
    'pca__n_components':Integer(10, 300),
    'clf__estimator__kernel': Categorical(['linear', 'rbf', 'poly']),
    'clf__estimator__C': Real(1e-3, 1e+2, prior='log-uniform'),
    'clf__estimator__gamma': Real(1e-4, 1e+1, prior='log-uniform'),
    'clf__estimator__degree': Integer(2, 5)
}

opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_space,
    scoring='accuracy',
    n_iter=35,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

opt.fit(X_train, y_train)

dump(opt.best_estimator_, "mnist_svm_model.joblib")

y_pred = opt.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Best Parameters:", opt.best_params_)
print("Test Accuracy:", acc)
