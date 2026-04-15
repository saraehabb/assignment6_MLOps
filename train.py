import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
print("New change for pipeline test")
print("Running training test with commit trigger")

print("Starting training")

X = np.random.rand(100, 3)
y = (X.sum(axis=1) > 1.5).astype(int)

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
print(f"Training done. Accuracy: {acc:.2f}")
