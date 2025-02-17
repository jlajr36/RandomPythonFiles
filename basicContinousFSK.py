import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Parameters
baud_rate = 8  # Baud rate (bits per second)
fs = 10000     # Sampling frequency (Hz)
f0 = 1000      # Frequency for binary '0' (Hz)
f1 = 2000      # Frequency for binary '1' (Hz)

msg = "this is a different msg"
data = [int(bit) for char in msg for bit in format(ord(char), '08b')]

num_bits = len(data)

# Calculate bit duration and total duration
bit_duration = 1 / baud_rate  # Duration of each bit (seconds)
total_duration = num_bits * bit_duration  # Total duration of the signal

# Time array
t = np.arange(0, total_duration, 1/fs)

# Generate FSK signal with smooth transitions
fsk_signal = np.zeros(len(t))
for i, bit in enumerate(data):
    start_time = i * bit_duration
    end_time = (i + 1) * bit_duration
    start_index = int(start_time * fs)
    end_index = int(end_time * fs)
    
    if bit == 0:
        fsk_signal[start_index:end_index] = np.sin(2 * np.pi * f0 * t[start_index:end_index])
    else:
        # Create a transition from f0 to f1
        transition_duration = 0.1  # Duration of the transition (seconds)
        transition_samples = int(transition_duration * fs)
        
        if i > 0 and data[i-1] == 0:  # Transition from 0 to 1
            transition_start_index = start_index - transition_samples
            transition_end_index = start_index
            fsk_signal[transition_start_index:transition_end_index] = np.sin(
                2 * np.pi * f0 * t[transition_start_index:transition_end_index] * (1 - (np.arange(transition_samples) / transition_samples)) +
                2 * np.pi * f1 * (np.arange(transition_samples) / transition_samples)
            )
        
        fsk_signal[start_index:end_index] = np.sin(2 * np.pi * f1 * t[start_index:end_index])

# Normalize the signal to the range of int16
fsk_signal_normalized = np.int16((fsk_signal / np.max(np.abs(fsk_signal))) * 32767)

# Save to WAV file
write('fsk_signal.wav', fs, fsk_signal_normalized)

# Plot the FSK signal
plt.figure(figsize=(10, 4))
plt.plot(t, fsk_signal)
plt.title(f'Continuous FSK Wave with Smooth Transitions (Baud Rate = {baud_rate} bps)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, total_duration)
plt.ylim(-1.5, 1.5)
plt.show()
