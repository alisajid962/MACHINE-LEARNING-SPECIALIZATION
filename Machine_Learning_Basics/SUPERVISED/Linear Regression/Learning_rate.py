import numpy as np
def compute_cost(X,y,w,b):
    m=X.shape[0]
    predictions=np.dot(X,w)+b
    errors=predictions-y
    cost =(1/(2*m))*np.sum(errors**2)
    return cost

def gradient_descent_with_history(X,y,w,b,alpha,num_iters):
    m=X.shape[0]
    w=w.copy()
    
    
    j_history=[]
    for i in range(num_iters):
        predictions=np.dot(X,w)+b
        errors=predictions-y
        
        dj_dw=(1/m)*np.dot(X.T,errors)
        db_dw=(1/m)*np.sum(errors)
        
        w=w-alpha*dj_dw
        b=b-db_dw
        current_Cost=compute_cost(X,y,w,b)
        j_history.append(current_Cost)
    
    return w, b, j_history
    
    
X_train = np.array([
    [1200, 2],  # House 1
    [1500, 3],  # House 2
    [1800, 3],  # House 3
    [2400, 4]   # House 4
])
y_train = np.array([300, 400, 500, 650])
w_init = np.array([0.0, 0.0])
b_init = 0.0
alpha = 1.0e-7 
iterations = 100
print("Starting Gradient Descent...\n")
final_w, final_b, J_hist = gradient_descent_with_history(
    X_train, y_train, w_init, b_init, alpha, iterations
)
print("\n--- Final Results ---")
print(f"Optimized Weights: {final_w}")
print(f"Optimized Bias: {final_b:.4f}")
print(f"History______________________________Bro______________________{J_hist}")