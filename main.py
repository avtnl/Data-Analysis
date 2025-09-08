import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hello import main

# Call the main function from hello.py
main()

# Create a small dataset
np.random.seed(42)
data = {
    "x": np.arange(1, 11),
    "y": np.random.normal(loc=5, scale=2, size=10)
}
df = pd.DataFrame(data)

# Print the DataFrame to confirm pandas is working
print("DataFrame preview:")
print(df)

# Create a simple Seaborn scatter plot
sns.set(style="darkgrid")
sns.scatterplot(data=df, x="x", y="y")

# Show the plot to confirm matplotlib and seaborn are working
plt.title("Test Plot: Seaborn + Matplotlib")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()

# Do a simple test to confirm scikit-learn is working (4 steps)

# Generate synthetic classification data (step 1)
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Split into train and test sets (step 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model (step 3)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate (step 4)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
