
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm                                                         # Support Vector Machine
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head(), '\n')
print(diabetes_dataset.shape, '\n')
print(diabetes_dataset.describe(), '\n')

print(diabetes_dataset['Outcome'].value_counts(), '\n')                         # 0 represent non-diabetes and 1 represent diabetes
print(diabetes_dataset.groupby('Outcome').mean())


# Separating the Data and Labels -

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X, '\n')
print(Y, '\n')


# Data Standardization -

scaler = StandardScaler()
scaler.fit(X)
standardized_data =scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']


# Train Test Split -

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape,'\n')


# Training the Model Using SVM-

classifier = svm.SVC(kernel='linear')                                           # Creates a Support Vector Classifier (SVC) model with a linear kernel
classifier.fit(X_train, Y_train)                                                # Trains the SVM model using the training data


# MODEL EVALUATION -

# Accurancy Score on the Training Data -
X_train_prediction = classifier.predict(X_train)
training_data_accurancy = accuracy_score(X_train_prediction, Y_train)
print('Accurancy score of training Data :', training_data_accurancy, '\n')

# Accurancy Score on the Testing Data -
X_test_prediction = classifier.predict(X_test)
testing_data_accurancy = accuracy_score(X_test_prediction, Y_test)
print('Accurancy score of testing Data :', testing_data_accurancy, '\n')


# Making a PREDICTIVE Data -

input_data = (7,147,76,0,0,39.4,0.257,43)

# Changing the input data to numpy array -
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance -
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

# Standardize the input data -
std_data = scaler.transform(input_data_reshape)
print(std_data,'\n')

prediction = classifier.predict(std_data)
print(prediction, '\n')

if (prediction[0] == 0) :
    print('The person is non-diabetic')
else:
    print('The person is Diabetic')




''' How Everything Connects:

    1. Load diabetes.csv using pandas.
    2. Explore dataset using head(), shape, and describe().
    3. Separate X (features) and Y (labels).
    4. Standardize X to improve model performance.
    5. Split data into training (80%) and testing (20%).
    6. Train an SVM model on training data.
    7. Evaluate model accuracy using accuracy_score.
    8. Make a prediction for a new input data.        '''