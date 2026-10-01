import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings


warnings.filterwarnings('ignore')

Data=pd.read_csv('D:\Documents\Python\VS code\Projects\Customer_Modeling_RLProject\Churn_Modelling.csv')

print('---Data Information---')

print(Data.head())
print(Data.info())
print(Data.shape)
print('All data points :',len(Data))
print(Data.isnull().sum())


print('---Data summary report---')

print(Data.describe())

plt.figure(figsize=(25,25))
sns.heatmap(Data.corr(numeric_only=True),annot=True,cmap='coolwarm')
plt.title('Corr of all Features')
plt.savefig('Corr of all Features.png')
plt.show()


## Feature Enginniering
to_drop=['RowNumber','CustomerId','Surname']

Data.drop(columns=to_drop,inplace=True)

print(list(Data.columns))

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()

Data['Gender']=le.fit_transform(Data['Gender'])

encoder=OneHotEncoder(sparse_output=False)
encoded=encoder.fit_transform(Data[['Geography']])
encoded_DF=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(['Geography']))

numerical=Data[['CreditScore','Age','Tenure','Balance','NumOfProducts'
,'HasCrCard','IsActiveMember','EstimatedSalary']]

## Split Data
from sklearn.model_selection import train_test_split

x=pd.concat([encoded_DF,numerical,Data[['Gender']]],axis=1)
y=Data['Exited']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

## Scalerition All Data

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

## Creating Model

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

Model=Sequential([
    Dense(32,activation='relu',input_shape=(x_train.shape[1],)),
    Dense(1,activation='sigmoid')
])

Model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print('---Model Summary Report---')
print(Model.summary())

Training=Model.fit(x_train_scaled,y_train,epochs=100,batch_size=10,validation_split=0.2)

loss,metric=Model.evaluate(x_test_scaled,y_test)
print('LOSS:',loss)
print('Metric:',metric)


Prediction=Model.predict(x_test_scaled)
print('Prediction:',Prediction[:5])
print('Y_test:',y_test[:5])


architectures = [
    [32],
    [64, 32],
    [32, 16],
    [128, 64]
]

results=[]

for arc in architectures:
    model=Sequential()
    for nourn in arc:
        model.add(Dense(nourn,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer='adam',
    loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(x_train_scaled,y_train,epochs=100,
    batch_size=10,validation_split=0.2)

    print('---Model Summary Report---')
    print(Model.summary())

    predict=model.predict(x_test_scaled)

    loss,metric=model.evaluate(x_test_scaled,y_test)
    print('Architecture:', arc)
    print('LOSS:', loss)
    print('Accuracy:', metric)

    results.append({
        'Architecture': arc,
        'Loss': loss,
        'Accuracy': metric
    })

results=pd.DataFrame(results)
print(results.sort_values('Accuracy',ascending=False))

## best model is arc=32 , loss=0.34 , Accuracy=0.86 and is our model at Training in Line 75


Prediction=(Prediction>0.5).astype(int)
cm=confusion_matrix(y_test,Prediction)
print(cm)

plt.figure(figsize=(25,25))
sns.heatmap(cm,annot=True,fmt='d')
plt.title('CM_Matrix')
plt.savefig('CM_Matrix.png')
plt.show()

pickle.dump(Model,open('Model.pkl','wb'))
pickle.dump(scaler,open('Scaler.pkl','wb'))
pickle.dump(x.columns,open('Columns.pkl','wb'))

print('All Files Saved ')





