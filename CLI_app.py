import numpy as np
import soundfile as sf
import os
from augmentation import _augmentation


def _read_audio_to_numpy(file_path: str) -> np.ndarray:
    """
    Чтение wav файла в numpy.     
    Input file_path: путь к wav файлу.
    Output audio: Аудио считанное в numpy формат.
    Output sample_rate: Частота дискретизации считанного аудио.
    """
    audio, sample_rate = sf.read(file_path)
    if len(audio.shape) == 1:
        audio = audio[:, np.newaxis]
    audio = audio.T
    audio = audio.astype(np.float32)
    return audio, sample_rate


def _save_numpy_to_audio(augmented_audio, sample_rate) -> None:
    """
    Сохранение numpy с аугментированным аудио в wav формат.     
    Input augmented_audio: Аугментированным аудио.
    Input sample_rate: Частота дискретизации сохраняемого аудио.
    Output: None.
    """
    sf.write('augmented/augmented_audio' + '.wav', augmented_audio.T, sample_rate)


def _get_valid_file_path() -> str:
    """
    Запрос на адрес аудио для аугментации.     
    Input: None.
    Output: адрес файла для аугментации.
    """
    while True:
        file_path = input("Введите путь к файлу: ")
        if not os.path.isfile(file_path):
            print(f"{file_path} не существует.")
        else:
            return file_path
        
        
def main() -> None:
    """
    Запуск процесса аугментации.     
    Input: None.
    Output: None.
    """
    file_path = _get_valid_file_path()
    audio, sample_rate = _read_audio_to_numpy(file_path)
    augmented_audio = _augmentation(audio)
    _save_numpy_to_audio(augmented_audio, sample_rate)


if __name__ == "__main__":
    main()
