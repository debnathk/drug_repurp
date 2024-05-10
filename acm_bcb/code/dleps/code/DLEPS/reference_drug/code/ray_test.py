import numpy as np
import ray

# Initialize Ray
ray.init()

@ray.remote
def process_row(row_a, b):
    c = np.zeros((b.shape[0], b.shape[1]), dtype=int)
    
    for j in range(b.shape[0]):
        for m in range(row_a.shape[0]):
            for n in range(b.shape[1]):
                if row_a[m] == b[j][n]:
                    c[j][n] = 1
    
    return c

# Your original data
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([[1, 2, 3],
              [4, 5, 6]])

# Parallel processing using Ray
clist = ray.get([process_row.remote(row_a, b) for row_a in a])

# Shutdown Ray
ray.shutdown()

# Convert the result to a NumPy array
result = np.array(clist)
print(result)
