#import required libraries
#pip install pandas
import pandas as pd
#pip install -U scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('diabetes.csv')
#if your ide could not load the dataset 
#try
#df = pd.read_csv(r'C:\path\diabetes.csv')

#Declare Target Variable
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

#split data into seperate trainig and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)

#instantiate the model
model = DecisionTreeClassifier()
#Train the model
model.fit(X_train, y_train)

#Evaulate the model
result = model.score(X_test, y_test)
#Print Accuracy
print("Accuracy: ", result)