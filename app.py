import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


st.set_page_config("Crop Recommendation System",layout = "wide")

st.title("Crop Recommendation System")
st.markdown("---")

df = pd.read_csv('Crop_recommendation.csv')

features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

acc = []
model = []

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')


from sklearn.model_selection import cross_val_score
score = cross_val_score(RF,features,target,cv=5)


import pickle
RF_pkl_filename = 'RandomForest.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

N = st.number_input('Enter The value of  Nitrogen')
P = st.number_input('Enter The value of  Phosphorous')
K = st.number_input('Enter the value of Potassium')
Temp = st.number_input('Enter the value of Temperature')
Hum = st.number_input('Enter the value of Humidity')
Ph = st.number_input('Enter the value of Ph')
rainfall = st.number_input('Enter the value of Rainfall')

selected_value = np.array([[N,P, K,Temp,Hum,Ph,rainfall]])
buttoon = st.button('Prediction')
if buttoon:
    
    st.write(RF.predict(selected_value))