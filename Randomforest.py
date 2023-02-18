# Random forest is also known as Ensemble learning is a supervised learning but multiple times we train the model
# It mathematical fromulation is like decision tree but here we take random nodes(Columns) to train for multiple times.
# Accuracy is higher compared to decision tree but the cons is large date may have computation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
data=pd.read_csv('diabetesfordisitiontree.csv')
x=data.drop('Outcome',axis=1).values
y=data['Outcome'].values
cl=RandomForestRegressor(n_estimators=20,random_state=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
cl.fit(x_train,y_train)
op=cl.predict(x_test)
accuracy=accuracy_score(y_test,op)
cr=classification_report(y_test,op)
print(accuracy)
print(cr)