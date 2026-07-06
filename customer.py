import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("bank.csv")
print(df.head())

print(df.info())
print(df.shape)

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


X = df.drop('y',axis=1)
y = df['y']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=42,random_state=42)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("accuracy_score:",accuracy_score(y_test,prediction))
print("classification_report:",classification_report(y_test,prediction))
print("confusion_matrix:\n",confusion_matrix(y_test,prediction))

new_customer = [[35,1,2,0,2000,1,0,0,15,5,200,1,-1,0,2,1]]
prediction = model.predict(new_customer)
print(prediction)

plt.figure(figsize=(20,10))

tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True,
    rounded=True
)

plt.show()

