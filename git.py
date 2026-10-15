#pip install mlflow
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
x,y = iris.data,iris.target
x_tr,x_te,y_tr,y_te=train_test_split(x,y,test_size=0.2,random_state=42)
models = {"Logistic Regression" : LogisticRegression(), "Decision Tree": DecisionTreeClassifier(), "Random Forest" : RandomForestClassifier()}
mlflow.set_experiment("Iris Experiment")

reg = "Iris_Classifier"
for name, model in models.items():
    with mlflow.start_run(nested=True):
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        acc = accuracy_score(y_te, pred)
        mlflow.log_param("Model", name)
        mlflow.log_metric("Accuracy", acc)
        mlflow.sklearn.log_model(sk_model = model, name = "model", registered_model_name=reg , input_example= x_tr[:5])
        print("Model:",name)
        print("Accuracy:",acc)
print("Experiment tracking  and model versioning completed")