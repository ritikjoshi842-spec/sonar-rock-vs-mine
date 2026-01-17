import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
file_path = '/content/drive/MyDrive/ROCK_OR_MINE.csv'
df = pd.read_csv(file_path, header= None)
df.head()
x= df.iloc[:,:-1]
y= df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model= LogisticRegression()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
accuracy= accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

