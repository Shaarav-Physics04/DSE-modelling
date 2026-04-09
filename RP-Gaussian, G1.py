import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Spatial domain
x = np.linspace(-20, 20, 4000)
dx = x[1] - x[0]

# Time domain
t = 0.2  # Time after which overlap occurs

# Constants
hbar = 1
m = 1

# Define Lorentzian wave packet
def lorentzian_packet(x, x0, gamma, k):
    lorentz_envelope = 1 / ((x - x0)**2 + gamma**2)
    return lorentz_envelope * np.exp(1j * k * (x - x0))

# Parameters
gamma = 0.9
d = 6.0
k = 14.0

# Superposition of two Lorentzian wave packets
psi0 = lorentzian_packet(x, -d/2, gamma, k) + lorentzian_packet(x, d/2, gamma, -k)

# Normalize the initial wavefunction
psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

# TDSE evolution using FFT (free particle propagation)
def evolve_tdse(psi0, x, t, hbar=1, m=1):
    N = len(x)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    psi_k = np.fft.fft(psi0)
    evolution_factor = np.exp(-1j * (hbar * k**2) * t / (2 * m))
    psi_t = np.fft.ifft(psi_k * evolution_factor)
    return psi_t

# Evolve the wavefunction
psi_t = evolve_tdse(psi0, x, t)
intensity = np.abs(psi_t)**2

# Detect peaks
peaks, _ = find_peaks(intensity, height=0.01, distance=20)
peak_count = len(peaks)

# Use only peak heights to compute visibility
if len(peaks) >= 2:
    peak_heights = intensity[peaks]
    Imax = np.max(peak_heights)
    Imin = np.min(peak_heights)
    visibility = (Imax - Imin) / (Imax + Imin)
else:
    Imax = Imin = visibility = 0

# Plot the result
plt.figure(figsize=(12, 4))
plt.plot(x, intensity, label='Intensity |ψ(x, t)|²')
plt.plot(x[peaks], intensity[peaks], "x", label=f'Peaks: {peak_count}', color='orange')
plt.axhline(Imax, color='red', linestyle='--', label=f'Imax ≈ {Imax:.3f}')
plt.axhline(Imin, color='blue', linestyle='--', label=f'Imin ≈ {Imin:.3f}')
plt.title(f'TDSE Lorentzian Interference (t = {t})\nFringe Count: {peak_count}, Visibility ≈ {visibility:.3f}')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



