import pandas as pd
##from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("diabetesfordisitiontree.csv")
print(data.shape)
print(data.head())
feature_values=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
x=data[feature_values]
y=data.Outcome

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
a=RandomForestRegressor(n_estimators=20,random_state=1)
a=a.fit(x_train,y_train)
y_pred=a.predict(x_test)
print(y_pred)
print(y_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred.round()))
