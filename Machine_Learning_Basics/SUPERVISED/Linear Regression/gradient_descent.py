import numpy as np

def compute_cost(x, y, w, b):
    m = len(x) # Number of training examples
    
    # f(x) = w*x + b
    predictions = (w * x) + b 
    
    # Calculate the squared errors for every point
    errors = predictions - y
    squared_errors = errors ** 2
    
    # Sum them up and divide by 2m
    total_cost = (1 / (2 * m)) * np.sum(squared_errors)
    
    return total_cost

def compute_gradient(x, y, w, b):
    m = len(x)
    
    predictions = (w * x) + b
    errors = predictions - y
    
    # Derivative for weight (dj_dw): average of (errors * x)
    dj_dw = (1 / m) * np.sum(errors * x)
    
    # Derivative for bias (dj_db): average of (errors)
    dj_db = (1 / m) * np.sum(errors)
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    cost_history = [] # We save the cost at every step to check for convergence
    
    for i in range(num_iters):
        # 1. Get the current slopes
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # 2. Update parameters (The core equation)
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        
        # 3. Calculate and record the new cost
        current_cost = compute_cost(x, y, w, b)
        cost_history.append(current_cost)
        
        # Print progress every 10% of the iterations
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost = {current_cost:.4f} | w = {w:.4f} | b = {b:.4f}")
            
    return w, b, cost_history


# Dummy House Data: 1000sqft = $300k, 2000sqft = $500k, 3000sqft = $700k
# A perfect fit would be w = 2, b = 1
x_train = np.array([1.0, 2.0, 3.0]) 
y_train = np.array([3.0, 5.0, 7.0])

# Initial guesses
initial_w = 0.0
initial_b = 0.0

# Hyperparameters
learning_rate = 0.1
iterations = 100

print("Starting Gradient Descent...")
final_w, final_b, costs = gradient_descent(
    x_train, y_train, 
    initial_w, initial_b, 
    learning_rate, iterations
)

print("\n--- Final Results ---")
print(f"Optimal Weight (w): {final_w:.4f}")
print(f"Optimal Bias (b): {final_b:.4f}")