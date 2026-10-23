import numpy as np
import time 

start_time_unvectorized=time.time()
n=100000
w=np.random.rand(n)
x=np.random.rand(n)
b=4.0
f_unvectorized=0
for j in range(n):
    f_unvectorized+=w[j]*x[j]
f_unvectorized+=b
unvectorized_time=(time.time()-start_time_unvectorized)*1000

start_time=time.time()
f_vectorized=np.dot(w,x)+b
vectorized_time=(time.time()-start_time)

print("Unvectorized_output:  ",f_unvectorized)
print(f"Time Taken for Unvectorized:   {unvectorized_time} ms")
print("=============================================")
print("Vectorized_output: ",f_vectorized)
print(f"Time taken for vectorized: {vectorized_time} ms")


# ====================================
# Unvectorized_output:   24990.744174897798
# Time Taken for Unvectorized:   84.39254760742188 ms
# =============================================
# Vectorized_output:  24990.744174897918
# Time taken for vectorized: 0.0003325939178466797 ms