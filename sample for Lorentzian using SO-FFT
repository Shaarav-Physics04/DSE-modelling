import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------------
# Spatial domain
# -------------------------------
x = np.linspace(-20, 20, 4000)
dx = x[1] - x[0]

# -------------------------------
# Time parameter
# -------------------------------
t = 0.2  # evolution time

# -------------------------------
# Constants
# -------------------------------
hbar = 1
m = 1

# -------------------------------
# Define Lorentzian wave packet
# -------------------------------
def lorentzian_packet(x, x0, gamma, k0):
    envelope = 1 / ((x - x0)**2 + gamma**2)
    return envelope * np.exp(1j * k0 * (x - x0))

# -------------------------------
# Parameters
# -------------------------------
gamma = 0.9   # width
d = 6.0       # separation
k0 = 14.0     # wave number

# -------------------------------
# Initial wavefunction (superposition)
# -------------------------------
psi0 = (
    lorentzian_packet(x, -d/2, gamma, k0) +
    lorentzian_packet(x,  d/2, gamma, -k0)
)

# Normalize initial wavefunction
psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, x))

# -------------------------------
# SO-FFT Evolution (Free Particle TDSE)
# -------------------------------
def evolve_sofft(psi0, x, t, hbar=1, m=1):
    N = len(x)
    dx = x[1] - x[0]

    # Momentum space grid
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    # FFT → momentum space
    psi_k = np.fft.fft(psi0)

    # Apply kinetic evolution operator
    evolution = np.exp(-1j * (hbar * k**2) * t / (2 * m))

    # Inverse FFT → position space
    psi_t = np.fft.ifft(psi_k * evolution)

    return psi_t

# -------------------------------
# Time evolution
# -------------------------------
psi_t = evolve_sofft(psi0, x, t)
intensity = np.abs(psi_t)**2

# -------------------------------
# Peak detection
# -------------------------------
peaks, _ = find_peaks(intensity, height=0.01, distance=20)
peak_count = len(peaks)

# -------------------------------
# Visibility (using peak heights ONLY)
# -------------------------------
if peak_count >= 2:
    peak_heights = intensity[peaks]
    Imax = np.max(peak_heights)
    Imin = np.min(peak_heights)
    visibility = (Imax - Imin) / (Imax + Imin)
else:
    Imax = Imin = visibility = 0

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(12, 4))
plt.plot(x, intensity, label='Intensity |ψ(x,t)|²')
plt.plot(x[peaks], intensity[peaks], "x", label=f'Peaks: {peak_count}')

plt.axhline(Imax, linestyle='--', label=f'Imax ≈ {Imax:.3f}')
plt.axhline(Imin, linestyle='--', label=f'Imin ≈ {Imin:.3f}')

plt.title(f'SO-FFT TDSE Lorentzian Interference (t = {t})\n'
          f'Fringe Count: {peak_count}, Visibility ≈ {visibility:.3f}')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
