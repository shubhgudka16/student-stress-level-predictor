import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 🔹 Load dataset
df = pd.read_csv("stress.csv")

# 🔹 Convert labels
df['stress_level'] = df['stress_level'].map({
    'Low': 0,
    'Medium': 1,
    'High': 2
})

# 🔹 Features & Target
X = df[['study_hours', 'sleep_hours', 'screen_time', 'exam_pressure']]
y = df['stress_level']


scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🔹 Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 🔹 Prediction
y_pred = model.predict(X_test)

# 🔹 Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 User Input
study = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))
screen = float(input("Enter screen time: "))
exam = float(input("Enter exam pressure (1-10): "))

user_data = [[study, sleep, screen, exam]]
prediction = model.predict(user_data)

if prediction[0] == 0:
    print("Stress Level: Low 😌")
elif prediction[0] == 1:
    print("Stress Level: Medium 😐")
else:
    print("Stress Level: High 😓")