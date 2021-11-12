import pandas as pd
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


MyDataColumns = ["E801_1A","E801_2A","E801_3A","E801_4A", "E801_5A", "E801_6A","E801_7A",
                 "E801_8A", "E801_9A", "E801_10A", "E801_11A","E801_12A", "E801_13A",
                 "E801_14A", "E801_15A","E801_16A", "E801_17A", "E801_18A","E801_19A",
                 "E801_20A", "E801_21A","E801_22A", "E801_23A", "E801_24A", "I01"]

dataset = pd.read_spss("SefSec_2014_Roster_weight new.sav", usecols=MyDataColumns)
DataAfterDeleteEmptyRow = dataset.dropna(axis=0, how='any')

x = DataAfterDeleteEmptyRow.iloc[:, 0:24].values
y = DataAfterDeleteEmptyRow.iloc[:, 24].values

label_encoder = LabelEncoder()
y_train=label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)

classifier=DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
Predict =classifier.predict(X_test)

confusion_matrix =str( confusion_matrix(y_test, Predict))
print(confusion_matrix+"\n")
summery=classification_report(y_test,Predict)
print(summery)




