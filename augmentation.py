import numpy as np
from scipy.signal import resample
from scipy.fft import fft, ifft
import warnings


# Гиперпараметры для аугментаций
speed_change = 0.9
white_noise_std = 0.01 
reverb_sound_change = 2
reverb_sound_shift = 5000
reverbation_times = 5
_delete_freq_bundaries = [0.8, 0.85]


def _add_reverbation(audio: np.ndarray, sound_change: int) -> np.ndarray:
    """
    Добавление ревербации в аудио сигнал.
    Input audio: Входной аудио сигнал.
    Input sound_change: коэфициент снижения громкости в ревербации.
    Output: Аудио сигнал с добавлением ревербации.
    """
    _, num_samples = audio.shape
    if num_samples < reverb_sound_shift:
        warnings.warn("Warning: the audio is too short for the _add_reverbation.")
        return audio
    reverb_audio = np.zeros_like(audio, dtype=np.float32, order='F')
    reverb_audio += audio

    shifted_audio = np.zeros_like(audio, dtype=np.float32, order='F')
    shifted_audio += audio
    for _ in range(reverbation_times):
        shifted_audio[:, reverb_sound_shift:] = shifted_audio[:, :-reverb_sound_shift]
        shifted_audio = shifted_audio / sound_change
        reverb_audio += shifted_audio
    return reverb_audio


def _add_white_noice(audio: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Добавление белого шума в аудио сигнал.
    Input audio: Входной аудио сигнал.
    Input frequency_factor: std генерируемого белого шума.
    Output: Аудио сигнал с добавлением белого шума.
    """
    noise = np.random.normal(0, noise_std, audio.shape).astype(np.float32, order='F')
    noisy_audio = audio + noise
    return noisy_audio


def _change_speed(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    """
    Изменение скорости аудио сигнала.
    Input audio: Входной аудио сигнал.
    Input speed_factor: Коэффициент изменения скорости.
    Output: Аудио сигнал с измененной скоростью.
    """
    num_channels, num_samples = audio.shape
    new_num_samples = int(num_samples / speed_factor)
    resampled_audio = np.empty((num_channels, new_num_samples), dtype=audio.dtype, order='F')
    resampled_audio[:, :] = resample(audio[:, :], new_num_samples, axis=1)
    return resampled_audio


def _delete_freq(audio: np.ndarray, freq_bundaries: list[float]) -> np.ndarray: # поскольку программа получает только numpy array, тут нет частоты дискретизации. Используем относительные границы.
    """
    Удаление участка частот.
    Input audio: Входной аудио сигнал.
    Input freq_borders: Относительные границы удаленного участка.
    Output: Аудио сигнал без участка частот.
    """
    modified_audio = np.copy(audio)
    _, num_samples = modified_audio.shape
    fft_spectrum = fft(modified_audio)
    start_index = int((num_samples - 1) // 2 * freq_bundaries[0])
    end_index = int((num_samples - 1) // 2 * freq_bundaries[1])
    fft_spectrum[:, start_index:end_index] = 0 + 0j
    neg_start_index = num_samples - end_index
    neg_end_index = num_samples - start_index
    fft_spectrum[:, neg_start_index:neg_end_index] = 0 + 0j
    modified_audio = np.real(ifft(fft_spectrum)).astype(np.float32, order='F')
    return modified_audio


def _augmentation(audio: np.ndarray) -> list:
    """
    Аугментации: изменение скорости аудио сигнала, добавление белого шума, добавление ревербации, удаление участка частот.
    Input audio: Входной аудио сигнал.
    Output: Аугментированный аудио сигнал.
    """
    augmented_audio = _change_speed(audio, speed_change)
    augmented_audio = _add_white_noice(augmented_audio, white_noise_std)
    augmented_audio = _add_reverbation(augmented_audio, reverb_sound_change)
    augmented_audio = _delete_freq(augmented_audio, _delete_freq_bundaries)
    return augmented_audio
