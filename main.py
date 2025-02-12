import numpy as np
import random

# --- Optimized Polynomial Arithmetic Functions ---

def cyclic_multiply(poly1, poly2, q):
    """
    Multiply two polynomials (stored as NumPy arrays) in Z_q[x]/(x^n-1)
    using full convolution and a vectorized wrap-around.
    
    Parameters:
      poly1, poly2: NumPy arrays of shape (n,)
      q: modulus
      
    Returns:
      A NumPy array of shape (n,) representing the product modulo (x^n-1, q)
    """
    n = poly1.shape[0]
    conv = np.convolve(poly1, poly2)  # Full convolution; length = 2*n - 1.
    # Wrap around: add the tail (indices n: end) to the beginning.
    conv[:n-1] += conv[n:]
    return conv[:n] % q

def poly_add(a, b, q):
    """Vectorized polynomial addition modulo q."""
    return (a + b) % q

def poly_sub(a, b, q):
    """Vectorized polynomial subtraction modulo q."""
    return (a - b) % q

def discrete_gaussian_noise(n, sigma=1.0):
    """
    Generate a NumPy array of length n with noise values drawn from a discrete Gaussian.
    
    This is done by sampling from a continuous Gaussian with standard deviation sigma,
    then rounding to the nearest integer.
    """
    samples = np.random.normal(0, sigma, n)
    return np.rint(samples).astype(int)

def center_coeffs(arr, q):
    """
    Center the coefficients of arr (which are modulo q) into the symmetric range 
    [-q//2, q//2] in a fully vectorized manner.
    """
    return np.where(arr > q//2, arr - q, arr)

# --- Parameter Setup for n = 1024, q = 2^23 - 1 ---

n = 1024                         # Dimension of the polynomial (one coefficient per bit)
q = (2**23) - 1                  # q = 8388607, a prime number of the form 2^23-1
message_scaling = q // 2         # Encode bit 1 as 4194303 and 0 as 0
sigma_noise = 1.0                # Standard deviation for our discrete Gaussian noise

# --- Key Generation ---

# Secret key s and error e: generated from a discrete Gaussian distribution.
s = discrete_gaussian_noise(n, sigma_noise)
e = discrete_gaussian_noise(n, sigma_noise)

# Public polynomial a: random coefficients uniformly drawn from [0, q).
a = np.random.randint(0, q, size=n)

# Compute public key b = a * s + e in Z_q[x]/(x^n-1).
b = poly_add(cyclic_multiply(a, s, q), e, q)

# --- Message Preparation ---

# Generate a random 1024-bit integer and convert it to a 1024-bit binary string.
message_int = random.getrandbits(1024)
message_bin = format(message_int, '01024b')  # Ensures a 1024-character string.
# Convert the binary string to a NumPy array of bits (0 or 1).
m = np.array([int(bit) for bit in message_bin], dtype=np.int32)
# Scale the message so that 1 is encoded as message_scaling (4194303) and 0 remains 0.
m_scaled = m * message_scaling

# --- Encryption ---

# Generate encryption randomness and noise from the discrete Gaussian.
e1 = discrete_gaussian_noise(n, sigma_noise)
e2 = discrete_gaussian_noise(n, sigma_noise)
r  = discrete_gaussian_noise(n, sigma_noise)  # Random polynomial for encryption.

# Compute:
#   u = a * r + e1,
#   v = b * r + e2 + m_scaled.
u = poly_add(cyclic_multiply(a, r, q), e1, q)
v = poly_add(poly_add(cyclic_multiply(b, r, q), e2, q), m_scaled, q)

# --- Decryption ---

# Compute md = v - s * u. In an ideal (noise-free) scenario, this equals m_scaled.
md = poly_sub(v, cyclic_multiply(s, u, q), q)
# Center the coefficients into the interval [-q//2, q//2].
centered_md = center_coeffs(md, q)
# Recover each bit by thresholding:
# If |coefficient| < message_scaling/2, decode as 0; otherwise, as 1.
recovered = np.where(np.abs(centered_md) < message_scaling / 2, 0, 1)
# Convert the recovered bit array back into an integer.
recovered_int = int("".join(recovered.astype(str)), 2)

# --- Output the Result ---

print("Recovered integer message (hex):", hex(recovered_int))
print((m==recovered).all())
