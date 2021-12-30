# Work 1 For Acoustic Signal Recognition


# Dependencies to be imported

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Predefined constants for default arguments

PREEMPHASIS_FACTOR = 0.97  # default factor for pre-emphasis
FRAME_SIZE_MS = 25  # default frame size (microsecond)
FRAME_SHIFT_MS = 10  # default frame shift (microsecond)
WINDOWING_FACTOR = 0.5  # default factor for windowing
MEL_FILTERS = 26  # default number of mel filters to be used
MEL_FILTER_LOW_FREQ = 20  # default lower bound of frequency to be adopted in the mel filter bank
MEL_FILTER_HIGH_FREQ = 6000  # default higher bound of frequency to be adopted in the mel filter bank
MFCC_NUMBERS = 13  # default number of MFCCs to be output

VISUALIZE_SUBPLOT_COLS = 3  # Columns of subplots
VISUALIZE_SUBPLOT_ROWS = 3  # Rows of subplots


def visualize_subplot(subplot_id):
    """
    Define the subplot to be shown
    :param subplot_id: (int) location of the subplot
    """
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.subplot(VISUALIZE_SUBPLOT_ROWS, VISUALIZE_SUBPLOT_COLS, subplot_id + 1)


def audio_read(filename, normalize=False):
    """
    Read a wave sequence from a certain file.
    :param normalize: (boolean) if the value is True, the audio signal will be normalized and scaled to [-1,1]
    :param filename: (str) path of the audio file
    :return: (ndarray) scaled amplitude of the audio file
    """
    sampling_rate, audio_signal = wavfile.read(filename)
    #print(sampling_rate, len(audio_signal))
    if normalize:
        audio_signal = audio_signal / 32767
    # Convert to mono-channel sequence
    if np.array(audio_signal.shape).shape[0] == 2:
        print("WARNING:The audio will be converted into a mono-channel sequence")
        audio_signal_mono = np.zeros((audio_signal.shape[0]))
        size = audio_signal.shape[0]
        channels = audio_signal.shape[1]
        for i in range(size):
            if i % 100000 == 0:
                print("Conversion progress:" + str(i / size * 100) + "%")
            for k in range(channels):
                audio_signal_mono[i] += audio_signal[i, k] / channels
        audio_signal = audio_signal_mono
    #print("Audio Read: done")
    return sampling_rate, audio_signal


def audio_preemphasis(audio_signal, preemphasis_factor=PREEMPHASIS_FACTOR):
    """
    Apply the pre-emphasis transform on an audio sequence
    :param audio_signal: (ndarray) input audio sequence
    :param preemphasis_factor: (float) factor of pre-emphasis
    :return: (ndarray) pre-emphasized audio sequence
    """
    audio_shift = np.concatenate((audio_signal[1:len(audio_signal)], [0]), 0)
    audio_preemphasized = audio_shift - audio_signal * preemphasis_factor
    #print("Audio Pre-emphasis: done")
    return audio_preemphasized, audio_signal


def audio_windowing(audio_signal, sampling_rate, windowing_factor=WINDOWING_FACTOR, frame_shift_ms=FRAME_SHIFT_MS,
                    frame_size_ms=FRAME_SIZE_MS):
    """
    Divide the audio sequence into tiny windows.
    :param audio_signal: (ndarray,1d) input audio sequence
    :param sampling_rate: (int/float) the sample frequence of the input audio sequence
    :param windowing_factor: (float) the weight factor used in Hanning or Hamming window
    :param frame_shift_ms: (int) the increment of the start point between two adjacent window
    :param frame_size_ms: (int) the length of each window
    :return: (ndarray,2d) windowed audio sequence
    """
    audio_len = len(audio_signal)
    frame_size = int(frame_size_ms * sampling_rate / 1000)
    frame_shift = int(frame_shift_ms * sampling_rate / 1000)
    audio_paddings = (frame_size - audio_len) % frame_shift
    audio_signal_padded = np.pad(audio_signal, (0, audio_paddings), mode="constant", constant_values=(0, 0))
    frame_count = (len(audio_signal_padded) - frame_size) // frame_shift + 1
    frame_window = np.zeros((frame_count, frame_size))
    #print("AUDIO LEN")
    #print(len(audio_signal_padded))
    for i in range(frame_count):
        for j in range(frame_shift * i, frame_shift * i + frame_size):
            t = j - frame_shift * i
            frame_window[i, t] = audio_signal_padded[j] * (
                    (1 - windowing_factor) - windowing_factor * np.cos(2 * np.pi * t / (frame_size - 1)))
    frame_window_transformed = frame_window.copy()
    #print("Windowing: done")
    return frame_window_transformed, frame_count, frame_size


def audio_fft(frame_window, frame_count,fixed_fft = 0):
    """
    Apply the STFT on the audio sequence
    :param frame_window: (ndarray,2d) input windowed audio sequence
    :param frame_count: (int) count of windowed frames
    :return: (ndarray,2d) spectrogram of the audio sequence
    """
    frame_window_transformed = []
    fft_bins = 512
    if fixed_fft == 0:
        # Makes the bin of FFT be the power of 2
        while True:
            if fft_bins < frame_window.shape[1]:
                fft_bins = fft_bins * 2
            else:
                break
    else:
        fft_bins = fixed_fft
    #fft_bins = frame_window.shape[1]
    # Perform FFT
    for i in range(frame_count):
        freq_amp = np.fft.rfft(frame_window[i], fft_bins)
        freq_amp_real = np.real(freq_amp)
        freq_amp_imag = np.imag(freq_amp)
        freq_amp_dist = (freq_amp_imag * freq_amp_imag + freq_amp_real * freq_amp_real)
        frame_window_transformed.append(freq_amp_dist)
    #print("DFT: done")
    return np.array(frame_window_transformed), fft_bins


def audio_spectrogram_visualize(frame_window_transformed, frame_size, sampling_freq, subplot_id=2):
    """
    Visualize the spectrogram
    :param subplot_id: (int) subplot location to be shown
    :param frame_window_transformed: (ndarray,2d) DFT-transformed matrix of audio data
    :param frame_size: (int) size of windowing frames
    :param sampling_freq: (int) sampling frequency of the original audio
    """
    # Preparing for the coordinates
    frame_window_trans_crop = frame_window_transformed[:, :frame_window_transformed.shape[1]]
    yt = np.linspace(0, frame_size, 10)
    xt = np.linspace(0, frame_window_transformed.shape[0], 10)
    s = sampling_freq // 2 / frame_size * yt
    r = FRAME_SHIFT_MS * np.linspace(0, frame_window_transformed.shape[0], 10)
    r = r.astype("int")
    s = s.astype("int")
    # The following conversion transforms the unit to decibel
    test = np.log10(frame_window_trans_crop.T + np.finfo("float").eps) * 10
    visualize_subplot(subplot_id)
    plt.pcolormesh(test)
    plt.yticks(yt, s)
    plt.ylabel("Frequency (Hz)")
    plt.xticks(xt, r)
    plt.xlabel("Time (ms)")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Decibel (dB)")
    plt.title("Acoustic Spectrogram")


def audio_frequency_visualize(audio_sequence, sampling_rate, title="Frequency Components of Audio Sequence"):
    """
    Visualize frequency components of an audio segment
    :param audio_sequence: (ndarray) audio segment
    :param sampling_rate: (int) sampling rate
    :param title: (str) title to be shown
    """
    freq_amp = np.abs(np.fft.rfft(audio_sequence, 512)) ** 2
    fftbins = 512
    st = [sampling_rate // 2 / fftbins * i for i in range(freq_amp.shape[0])]
    visualize_subplot(0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Accumulated Energy (dB)")
    plt.title(title)
    freq_amp = np.log10(freq_amp + np.finfo("float").eps) * 10
    plt.plot(st, freq_amp)


def mel_freq(normal_freq):
    """
    Get the Mel frequency according to the given frequency
    :param normal_freq:(ndarray) Input frequency
    :return:(ndarray) Mel frequency corresponding to the input
    """
    return 1125 * np.log(1 + normal_freq / 700)


def mel_freq_inverse(m_freq):
    """
    The inverse function of Mel frequency function
    :param m_freq: (ndarray) Mel frequency
    :return: (ndarray) Original frequency
    """
    return 700 * (np.exp(m_freq / 1125) - 1)


def get_mel_filter_banks(sampling_freq, fft_size, filters=MEL_FILTERS, low_freq=MEL_FILTER_LOW_FREQ,
                         high_freq=MEL_FILTER_HIGH_FREQ):
    """
    Get the matrix for FFT-bins-aligned Mel filter bank matrix
    :param sampling_freq: (int) Sampling frequency
    :param fft_size: (int) Size of FFT bins
    :param filters:  (int) Size of output numbers of Mel filters
    :param low_freq: (int) the lower bound frequency of Mel filters
    :param high_freq:  (int) the higher bound frequency of Mel filters
    :return: (ndarray,2d) Filter matrix, each row corresponds to a Mel filter, each column corresponds to a FFT bin
    """
    m_freqs = np.array([mel_freq(low_freq) + i * (mel_freq(high_freq) - mel_freq(low_freq)) / (filters + 1) for i in
                        range(filters + 2)])
    norm_freqs = mel_freq_inverse(m_freqs)
    fft_aligned_freq = np.floor((fft_size / sampling_freq) * norm_freqs)
    mel_filters = np.zeros((filters, fft_size // 2 + 1))
    for i in range(filters):
        for j in range(fft_size):
            if fft_aligned_freq[i + 1] >= j > fft_aligned_freq[i]:
                mel_filters[i, j] = (j - fft_aligned_freq[i]) / (fft_aligned_freq[i + 1] - fft_aligned_freq[i])
            elif fft_aligned_freq[i + 1] < j < fft_aligned_freq[i + 2]:
                mel_filters[i, j] = (fft_aligned_freq[i + 2] - j) / (fft_aligned_freq[i + 2] - fft_aligned_freq[i + 1])
    #print("Get Mel-filter Bank: done")
    return mel_filters


def get_feat_and_energy(stft_matrix, mel_filter_banks, log_energy=True):
    """
    Get feature and energy from the acoustic signal processed by STFT
    :param log_energy: (boolean, optional) If the value is True, logarithmic energy will be used
    :param stft_matrix: (ndarray,2d) acoustic signal processed by STFT
    :param mel_filter_banks: (ndarray,2d) Mel filter bank matrix
    :return: (tuple) the first element is feature, the second one is energy
    """
    feat = np.dot(stft_matrix, mel_filter_banks.T)
    energy = np.sum(stft_matrix, 1)
    for i in range(len(energy)):
        if energy[i] == 0:
            energy[i] = np.finfo(float).eps
    if log_energy:
        energy = np.log(energy)
    #print("Calculate Feat (Apply Mel-filter Bank): done")
    return feat, energy


def visualize_mel_filter_bank(mel_filter_bank):
    """
    Visualize Mel filter bank via plot graph
    :param mel_filter_bank:(ndarray,2d) Mel filter bank matrix
    """
    fft_bins = mel_filter_bank.shape[1]
    mel_filters = mel_filter_bank.shape[0]
    x_axis = np.array([i for i in range(fft_bins)])
    visualize_subplot(3)
    plt.title("Mel Filter Banks")
    plt.xlabel("FFT bin")
    plt.ylabel("Scale Factor")
    for i in range(mel_filters):
        plt.plot(x_axis, mel_filter_bank[i, :])


def feat_spectrogram_visualize(feat):
    """
    Visualize feature via color map graph
    :param feat: (ndarray,2d) feature matrix
    """
    feat_crop = feat[:, :feat.shape[1]]
    xt = np.linspace(0, feat.shape[0], 10)
    r = FRAME_SHIFT_MS * np.linspace(0, feat.shape[0], 10)
    r = r.astype("int")
    test = feat_crop.T
    visualize_subplot(4)
    plt.pcolormesh(test)
    plt.ylabel("Mel-Filter Results")
    plt.xticks(xt, r)
    plt.xlabel("Time (ms)")
    plt.colorbar()
    plt.title("Feat Spectrogram")


def audio_dct(feat, mfcc_nums=MFCC_NUMBERS):
    """
    Perform the discrete cosine transformation(DCT) on a feature matrix
    :param feat: (ndarray,2d) feature matrix
    :param mfcc_nums: (int) output MFCC numbers
    :return: (ndarray,2d) DCT-processed feature matrix
    """
    dct_matrix = np.zeros((feat.shape[0], mfcc_nums))
    for i in range(feat.shape[0]):
        for j in range(mfcc_nums):
            for k in range(feat.shape[1]):
                dct_matrix[i, j] += feat[i, k] * np.cos((k + 0.5) * j * np.pi / feat.shape[1])
    #print("DCT: done")
    #print(dct_matrix.shape)
    return dct_matrix


def audio_dct_lib(feat, mfcc_nums=MFCC_NUMBERS):
    """
    Perform the discrete cosine transformation(DCT) on a feature matrix
    :param feat: (ndarray,2d) feature matrix
    :param mfcc_nums: (int) output MFCC numbers
    :return: (ndarray,2d) DCT-processed feature matrix
    """
    dct_matrix = np.zeros((feat.shape[0], mfcc_nums))
    import scipy.fftpack as fftpck
    dct_matrix = fftpck.dct(feat, type=2, axis=1, norm='ortho')[:,:mfcc_nums]
    return dct_matrix

def cepstrum_spectrogram_visualize(feat, subplot_id=5, title="Cepstrum Color Map"):
    """
    Visualize cepstrum via color map graph
    :param title: (str) title to be shown
    :param subplot_id: (int) subplot location
    :param feat:(ndarray,2d) cepstrum matrix
    """
    feat_crop = feat[:, :feat.shape[1]]
    xt = np.linspace(0, feat.shape[0], 5)
    r = FRAME_SHIFT_MS * np.linspace(0, feat.shape[0], 5)
    r = r.astype("int")
    test = feat_crop.T
    visualize_subplot(subplot_id)
    plt.pcolormesh(test)
    plt.ylabel("Cepstrum")
    plt.xticks(xt, r)
    plt.xlabel("Time (ms)")
    plt.colorbar()
    plt.title(title)


def feature_spectrogram_visualize(feat, subplot_id=5, title="Feature Map"):
    """
    Visualize cepstrum via color map graph
    :param title: (str) title to be shown
    :param subplot_id: (int) subplot location
    :param feat:(ndarray,2d) cepstrum matrix
    """
    feat_crop = feat[:, :feat.shape[1]]
    xt = np.linspace(0, feat.shape[0], 5)
    r = FRAME_SHIFT_MS * np.linspace(0, feat.shape[0], 5)
    r = r.astype("int")
    test = feat_crop.T
    visualize_subplot(subplot_id)
    plt.pcolormesh(test)
    plt.ylabel("Feature")
    plt.xticks(xt, r)
    plt.xlabel("Time (ms)")
    plt.colorbar()
    plt.title(title)


def audio_sequence_visualize(audio_sequence, title, subplot_id):
    """
    Visualize the audio sequence in a plot diagram
    :param subplot_id: (int) location of subplot
    :param audio_sequence: (ndarray,1d) audio sequence read
    :param title: (str) title to be shown in the graph
    """
    visualize_subplot(subplot_id)
    x_axis = np.array([i for i in range(len(audio_sequence))])
    plt.plot(x_axis, audio_sequence)
    plt.xlabel("Sample Points")
    plt.ylabel("Amplitude")
    plt.title(title)


def calculate_delta_feature(feat, n):
    """
    Calculate the dynamic mfcc feature
    :param feat: (ndarray,2d) feature
    :param n: (int) this should be 1 according to PPT
    :return: (ndarray,2d) dynamic feature
    """
    deno = 2 * np.sum([i * i for i in range(1, n + 1)])
    feat_padded = np.pad(feat, ((n, n), (0, 0)), mode="edge")
    d_feat = np.zeros_like(feat)
    weight = np.arange(-n, n + 1)
    for i in range(feat.shape[0]):
        d_feat[i] = np.dot(weight, feat_padded[i: i + 2 * n + 1]) / deno
    #print("Calculating Dynamic Feature: done")
    return d_feat


def calculate_delta_energy(energy, n):
    """
    Calculate the dynamic energy feature
    :param energy: (ndarray,1d) energy feature
    :param n: (int)
    :return: (ndarray,1d) dynamic energy feature
    """
    deno = 2 * np.sum([i * i for i in range(1, n + 1)])
    energy_padded = np.pad(energy, (n, n))
    d_energy = np.zeros_like(energy)
    weight = np.arange(-n, n + 1)
    for i in range(energy.shape[0]):
        d_energy[i] = np.dot(weight, energy_padded[i: i + 2 * n + 1]) / deno
    #print("Calculating Dynamic Energy: done")
    return d_energy


def audio_lifter(cepstra, L=22):
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def feature_concatenate_normalisation(feature,normalisation=True):
    """
    Concatenate vectors of features, and perform Cesptral Mean & Variance Normalization (CMVN)
    on the matrix. The normalization formula coincides with the one given in the PPT
    To specific the range and details of the normalization, following papers are referenced
     - Chen C P , Bilmes J , Kirchhoff K . Low-Resource Noise-Robust Feature Post-Processing On Aurora 2.0. 2002.
     - Chapaneri S V , Chapaneri S V . Spoken Digits Recognition using Weighted MFCC and Improved Features for Dynamic
       Time Warping[J]. International Journal of Computer Applications, 2012, 40(3):6-12.
    :param feature: (ndarray,tuple,list) MFCC feature or the delta feature
    :return: (ndarray) Concatenated matrix of MFCC feature. Each row contains a frame, and each column
     contains a CMVN-normalized feature
    """
    # First, concatenate features to generate the feature matrix
    if isinstance(feature, tuple) or isinstance(feature, list):
        feature_f = np.array(feature)
    else:
        feature_f = feature
    feature_shape = feature_f.shape
    numbers = feature_shape[0]
    concat_list = []
    for i in range(numbers):
        concat_list.append(feature_f[i])
    concat_tuple = tuple(concat_list)
    concat_matrix = np.concatenate(concat_tuple, axis=1)
    # Then, perform Cesptral Mean & Variance Normalization (CMVN) on each feature.
    if normalisation:
        feature_count = concat_matrix.shape[1]
        for i in range(feature_count):
            feature_mean = np.mean(concat_matrix[:, i])
            feature_std_var = np.std(concat_matrix[:, i], ddof=1)
            concat_matrix[:, i] = (concat_matrix[:, i] - feature_mean) / feature_std_var
    return concat_matrix


def main():
    # Read audio & Pre-emphasis
    sampling_freq, audio = audio_read("hello_1950641.wav", True)
    audio, audio_original = audio_preemphasis(audio)
    audio_sequence_visualize(audio_original, "Original Signal", 0)
    audio_sequence_visualize(audio, "Pre-emphasized Signal", 1)

    # Windowing (Padding Zero)
    frame, frame_count, frame_size = audio_windowing(audio, sampling_freq)

    # Short-time Fourier Transformation (STFT)
    frame_stft, fft_bins = audio_fft(frame, frame_count)
    audio_spectrogram_visualize(frame_stft, fft_bins // 2 + 1, sampling_freq)

    # Get Mel Filter Banks
    mel_filter_banks = get_mel_filter_banks(sampling_freq, fft_bins, MEL_FILTERS, 300, sampling_freq // 2)
    visualize_mel_filter_bank(mel_filter_banks)

    # Apply Mel Filter & Log
    feat, energy = get_feat_and_energy(frame_stft, mel_filter_banks)
    log_feat = np.log(feat)
    feat_spectrogram_visualize(log_feat)

    # Apply Discrete Cosine Transformation (DCT)
    dct_result = audio_dct(log_feat)
    dct_result[:, 0] = energy
    cepstrum_spectrogram_visualize(dct_result, 5, "MFCC Result")

    # Calculate Dynamic Features
    feat_order_1 = calculate_delta_feature(dct_result, 1)
    feat_order_2 = calculate_delta_feature(feat_order_1, 1)
    feature_spectrogram_visualize(feat_order_1, 6, "Dynamic Feature (Order 1)")
    feature_spectrogram_visualize(feat_order_2, 7, "Dynamic Feature (Order 2)")

    # Feature Normalization
    feature_vector = feature_concatenate_normalisation((dct_result, feat_order_1, feat_order_2))
    feature_spectrogram_visualize(feature_vector, 8, "Normalized Feature Matrix")

    # Process Done
    print("Process Finished")


if __name__ == "__main__":
    main()
