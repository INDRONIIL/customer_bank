import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("bank.csv")
print(df.head())

print(df.info())
print(df.shape)

X = df.drop('y',axis=1)
y = df['y']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=42,random_state=42)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
    criterion="entropy"
)
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("accuracy_score:",accuracy_score(y_test,prediction))
print("classification_report:",classification_report(y_test,prediction))
print("confusion_matrix:",confusion_matrix(y_test,prediction))