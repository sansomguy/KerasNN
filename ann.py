#%%

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, 13].values

#%%
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#%%
# Gender encoder
labelencoder_X_Gender = LabelEncoder()
X[:, 2] = labelencoder_X_Gender.fit_transform(X[:, 2])

#%%
labelencoder_X_Country = LabelEncoder()
X[:, 1] =  labelencoder_X_Country.fit_transform(X[:, 1])

#%% Creaet new columns for all the different country categories 
# (This is because 0, is not worth less than 2, when considering country ids)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#%% Avoid dummy variable trap
X = X[:,1:]

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#%% Build the model

def build_classifier(optimizer):
	classifier = Sequential()
	
	classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
	classifier.add(Dropout(rate=0.1))
	classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
	classifier.add(Dropout(rate=0.1))
	classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
	
	classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	
	return classifier

	
#%% Figure out the best hyper parameters for the network
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {
		'batch_size': [10,25,32], 
		'epochs': [100,200,300], 
		'optimizer': ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, cv=10, scoring='accuracy')

#%% Perform training
grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
best_estimator= grid_search.best_estimator_

#%% Inspect the variance and the mean
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=best_estimator, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

#%% Predicting and evaluating the results
y_pred = classifier.predict(X_test)

#%% Convert probabilities to binary
y_pred = (y_pred > 0.5)

#%% Assess the results
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



#%% Make a prediction concerning the following data
#Account
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# Does this customer have a credit card ? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $50000
account = [0,0, 600, 1, 40, 3, 60000,2, 1, 1, 50000]

# now we need to scale this value
account = sc.transform([account])
account_pred = classifier.predict(account)



