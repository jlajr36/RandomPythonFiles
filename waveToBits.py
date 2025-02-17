import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

# Read the FSK signal from the WAV file
fs, fsk_signal = read('fsk_signal.wav')

# Normalize the signal
fsk_signal = fsk_signal / np.max(np.abs(fsk_signal))

# Parameters
f0 = 1000  # Frequency for binary '0' (Hz)
f1 = 2000  # Frequency for binary '1' (Hz)
baud_rate = 8  # Baud rate (bits per second)
bit_duration = 1 / baud_rate  # Duration of each bit (seconds)
num_samples_per_bit = int(bit_duration * fs)  # Number of samples per bit

# Initialize an empty list to hold the decoded bits
decoded_bits = []

# Loop through the signal in chunks of num_samples_per_bit
for i in range(0, len(fsk_signal), num_samples_per_bit):
    # Get the current chunk of the signal
    chunk = fsk_signal[i:i + num_samples_per_bit]
    
    # Calculate the frequency of the chunk using FFT
    if len(chunk) == num_samples_per_bit:  # Ensure the chunk is the correct size
        freqs = np.fft.fftfreq(len(chunk), 1/fs)
        fft_magnitude = np.abs(np.fft.fft(chunk))
        
        # Find the peak frequency in the FFT
        peak_freq = freqs[np.argmax(fft_magnitude)]
        
        # Determine if the peak frequency corresponds to a '0' or '1'
        if abs(peak_freq - f0) < abs(peak_freq - f1):
            decoded_bits.append(0)
        else:
            decoded_bits.append(1)

# Print the decoded bits
print("Decoded bits:", decoded_bits)

chars = []
for i in range(0, len(decoded_bits), 8):
    byte = decoded_bits[i:i+8]  # Get the next 8 bits
    # Convert the byte (list of bits) to a character
    char = chr(int(''.join(map(str, byte)), 2))
    chars.append(char)
msg = ''.join(chars)

# Print the decoded msg
print("Decoded message:", msg)

# Optional: Plot the original FSK signal
t = np.arange(0, len(fsk_signal) / fs, 1/fs)
plt.figure(figsize=(10, 4))
plt.plot(t, fsk_signal)
plt.title('FSK Signal from WAV File')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(0, len(fsk_signal) / fs)
plt.ylim(-1.5, 1.5)
plt.show()
