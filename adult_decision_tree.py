import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Adult dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain",
    "capital-loss", "hours-per-week",
    "native-country", "income"
]

data = pd.read_csv(
    url,
    names=columns,
    na_values=" ?",
    skipinitialspace=True
)

print("First 5 records:")
print(data.head())

# Remove missing values
data = data.dropna()

# Convert text values into numbers
encoder = LabelEncoder()

for column in data.select_dtypes(include="object").columns:
    data[column] = encoder.fit_transform(data[column])


# Separate input and output
X = data.drop("income", axis=1)
y = data["income"]


# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Find accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
