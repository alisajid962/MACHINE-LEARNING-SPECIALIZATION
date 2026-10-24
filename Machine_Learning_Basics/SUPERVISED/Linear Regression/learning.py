import numpy as np 
from sklearn.linear_model import LinearRegression
X=np.array([[1],[2],[3],[4],[5]
            ])
# 1. Prepare the Data
# X represents our input feature (e.g., years of experience)
# Scikit-learn expects 2D arrays for inputs, hence the nested brackets
X = np.array([[1], [2], [3], [4], [5]]) 

# y represents the actual target values (e.g., salary in thousands)
y = np.array([15, 28, 31, 45, 49])


model=LinearRegression()
model.fit(X,y)

w=model.coef_[0]
b=model.intercept_

print(f"Learned Weight (w): {w:.2f}")
print(f"Learned Bias (b): {b:.2f}")

new_input=np.array([[0]])
prediction=model.predict(new_input)
print("predictions",prediction)
print("======================")

predictions=model.predict(X)
print(predictions)