# -*- coding: utf-8 -*-
"""
Created on Mon May  5 07:45:08 2025

@author: yt517
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import sounddevice as sd

# -------------------- Define the Global Variables --------------------

fs = 44100
duration = 3  
t = np.linspace(0, duration, duration * fs)
# -------------------- Helper Functions Section --------------------


# Function to Transform Time Domain Signal into the Frequncy Domain
def Frequency_Conversion(x, sr):
    N = sr
    X = fft(x)
    X_mag = np.abs(X[0:int(N // 2)])
    X_single = (2.0 / N) * X_mag
    f = np.linspace(0, fs/2, N//2, endpoint=False)
    return f, X_single

def Remove(Signal, fs):
    N = len(Signal)
    X = fft(Signal)

    # We only need to scan the positive‐frequency half
    half = N // 2

    # First removal
    maxim = 0.0
    idx = -1
    for i in range(half):
        mag = abs(X[i])
        if mag > maxim:
            maxim = mag
            idx = i
    X[idx] = 0
    X[-idx] = 0

    # Second removal
    maxim = 0.0
    idx = -1
    for i in range(half):
        mag = abs(X[i])
        if mag > maxim:
            maxim = mag
            idx = i
    X[idx] = 0
    X[-idx] = 0

    # Inverse FFT back to time domain
    x_rec = ifft(X)
    return np.real(x_rec)

# Hashmap of the different octave values for notes
Note_Freq_Array = np.array([
    [130.81, 261.63],  # C
    [146.83, 293.66],  # D
    [164.81, 329.63],  # E
    [174.61, 349.23],  # F
    [196.00, 392.00],  # G
    [220.00, 440.00],  # A
    [246.94, 493.88],  # B
])

#Song_Note, Start_Time, Duration
# Song Elements (A Bad Attempt Of Megalovania from Undertale)
Song_Elements_Time = [
    # Part 1
    [1, 0.000, 0.125],  # D
    [5, 0.125, 0.125],  # A
    [4, 0.250, 0.125],  # G♯ 
    [4, 0.375, 0.125],  # G
    [3, 0.500, 0.125],  # F
    [1, 0.625, 0.125],  # D
    [3, 0.750, 0.125],  # F
    [4, 0.875, 0.125],  # G

    # Part 2
    [1, 1.000, 0.125],  # D
    [5, 1.125, 0.125],  # A
    [4, 1.250, 0.125],  # G♯
    [4, 1.375, 0.125],  # G
    [3, 1.500, 0.125],  # F
    [1, 1.625, 0.125],  # D
    [3, 1.750, 0.125],  # F
    [4, 1.875, 0.125],  # G

    # Part 3
    [1, 2.000, 0.125],  # D
    [5, 2.125, 0.125],  # A
    [4, 2.250, 0.125],  # G♯
    [4, 2.375, 0.125],  # G
    [3, 2.500, 0.125],  # F
    [1, 2.625, 0.125],  # D
    [3, 2.750, 0.125],  # F
    [4, 2.875, 0.125],  # G
]

Base_Signal = np.zeros(t.shape, dtype=t.dtype)
for Song_Note, ti, Ti in Song_Elements_Time:
      f3 = Note_Freq_Array[Song_Note, 0]
      f4 = Note_Freq_Array[Song_Note, 1] 
      Effective_Time = (t >= ti) & (t < ti + Ti) 
      
      Base_Signal += (np.sin(2*np.pi*f3*t) + np.sin(2*np.pi*f4*t))*Effective_Time

# Adding Bad Noise :(
fn1, fn2 = np.random.randint(0, 512, 2)
Noise_Signal = np.sin(2*np.pi*fn1*t) + np.sin(2*np.pi*fn2*t)
Total_Signal = Base_Signal + Noise_Signal

# Remove Bad Noise :)
Filtered_Signal = Remove(Total_Signal, fs)

# prepare frequency-domain data
Clean_Freq, Fourier_Clean = Frequency_Conversion(Base_Signal, fs)
Noisy_Freq, Fourier_Noisy = Frequency_Conversion(Total_Signal, fs)
Filtered_Freq, Fourier_Filtered = Frequency_Conversion(Filtered_Signal, fs)

# -------------------- End Of Section --------------------


# -------------------- Output Section --------------------

# Songs Playback
# Original
print("Playing: Original Signal (Clean)")
sd.play(Base_Signal / np.max(np.abs(Base_Signal)), fs)
sd.wait()

# Noisy
print(f"Playing: Signal + Noise (fn1={fn1} Hz, fn2={fn2} Hz)")
sd.play(Total_Signal / np.max(np.abs(Total_Signal)), fs)
sd.wait()

# Filtered
print("Playing: Filtered Signal (Noise Removed)")
sd.play(Filtered_Signal / np.max(np.abs(Filtered_Signal)), fs)
sd.wait()

# Plotting: 2 rows x 3 cols
plt.figure(figsize=(12, 8))

# Time-domain plots
plt.subplot(2, 3, 1)
plt.plot(t, Base_Signal); plt.title("Time: Clean")
plt.subplot(2, 3, 2)
plt.plot(t, Total_Signal); plt.title("Time: Noisy")
plt.subplot(2, 3, 3)
plt.plot(t, Filtered_Signal); plt.title("Time: Filtered")

# Frequency-domain plots
plt.subplot(2, 3, 4)
plt.plot(Clean_Freq, Fourier_Clean); plt.title("Freq: Clean")
plt.subplot(2, 3, 5)
plt.plot(Noisy_Freq, Fourier_Noisy); plt.title("Freq: Noisy")
plt.subplot(2, 3, 6)
plt.plot(Filtered_Freq, Fourier_Filtered); plt.title("Freq: Filtered")

plt.tight_layout()
plt.show()
