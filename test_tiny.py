import numpy as np

# Test on a tiny known image first
test = np.array([[1,2,3,4],
                 [2,3,4,5],
                 [3,4,5,6]], dtype=np.float64)

gy_t, gx_t = np.gradient(test)
I_rt = poisson_reconstruct(gx_t, gy_t, float(test.mean()))

print("Test input:")
print(test)
print("Round-trip:")
print(np.round(I_rt, 4))
print("Max error:", np.abs(test - I_rt).max())
