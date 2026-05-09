#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import warnings
import hashlib
import pickle
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import skew
from tqdm import tqdm

# -------------------------------------------------------------------
#  Игнорируем несущественные предупреждения MNE
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

# -------------------------------------------------------------------
#  Функции обработки (те же, что были, но без print-ов для чистоты)
# -------------------------------------------------------------------

def load_psg_with_annotations(file_path: str):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        annotations = raw.annotations
        return raw, annotations
    except Exception as e:
        return None, None

def preprocess_eeg_only(raw: mne.io.Raw) -> Optional[mne.io.Raw]:
    raw_processed = raw.copy()
    eeg_channels = [
        'EEG F3-A2', 'EEG C3-A2', 'EEG O1-A2',
        'EEG F4-A1', 'EEG C4-A1', 'EEG O2-A1'
    ]
    available_channels = [ch for ch in eeg_channels if ch in raw_processed.ch_names]
    if not available_channels:
        return None
    for eeg_ch in available_channels:
        try:
            raw_processed.filter(l_freq=0.5, h_freq=45,
                                 picks=[eeg_ch], method='iir', verbose=False)
            raw_processed.notch_filter(50, picks=[eeg_ch], verbose=False)
        except Exception:
            continue
    return raw_processed

def map_annotation_to_epoch_fixed(epoch_start: float, epoch_end: float,
                                  annotations: mne.Annotations) -> Dict:
    epoch_annotations = {
        'sleep_stage': -1,
        'apnea_count': 0, 'hypopnea_count': 0, 'snore_count': 0,
        'desaturation_count': 0, 'tachycardia_count': 0,
        'leg_movement_count': 0, 'periodic_leg_movement_count': 0,
        'bruxism_count': 0, 'activation_count': 0,
        'k_complex_count': 0, 'sleep_spindle_count': 0, 'artefact_count': 0,
        'has_apnea': 0, 'has_hypopnea': 0, 'has_desaturation': 0,
        'has_snore': 0, 'has_bruxism': 0,
    }

    sleep_stage_map = {
        'Sleep stage W': 0, 'Sleep stage W(eventUnknown)': 0,
        'Sleep stage 1': 1, 'Sleep stage 1(eventUnknown)': 1,
        'Sleep stage 2': 2, 'Sleep stage 2(eventUnknown)': 2,
        'Sleep stage 3': 3, 'Sleep stage 3(eventUnknown)': 3,
        'Sleep stage R': 4, 'Sleep stage R(eventUnknown)': 4,
        'pointPolySomnographyREM': 4, 'БДГ': 4
    }
    event_type_map = {
        'pointPolySomnographyObstructiveApnea': 'apnea',
        'Obstructive апноэ': 'apnea',
        'pointPolySomnographyHypopnea': 'hypopnea',
        'pointPolySomnographySnore': 'snore',
        'pointPolySomnographyTachycardia': 'tachycardia',
        'pointPolySomnographyDesaturation': 'desaturation',
        'pointPolySomnographyLegsMovements': 'leg_movement',
        'pointPolySomnographyPeriodicalLegsMovements': 'periodic_leg_movement',
        'pointBruxism': 'bruxism', 'Бруксизм': 'bruxism',
        'pointPolySomnographyActivation': 'activation',
        'pointPolySomnographyK_complex': 'k_complex',
        'pointPolySomnographySleepSpindle': 'sleep_spindle',
        'blockArtefact': 'artefact'
    }

    for ann_start, ann_duration, ann_description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        ann_end = ann_start + ann_duration
        if not (ann_end < epoch_start or ann_start > epoch_end):
            desc = str(ann_description).lower()
            # стадии сна
            if 'бдг' in desc or 'pointpolysomnographyrem' in desc:
                epoch_annotations['sleep_stage'] = 4
            elif 'sleep stage' in desc:
                for stage_key, stage_code in sleep_stage_map.items():
                    if stage_key.lower() in desc:
                        epoch_annotations['sleep_stage'] = stage_code
                        break
            # события
            for event_key, event_type in event_type_map.items():
                if event_key.lower() in desc or (
                    event_type == 'apnea' and any(w in desc for w in ['апноэ', 'apnea'])
                ) or (event_type == 'bruxism' and any(w in desc for w in ['бруксизм', 'bruxism'])):
                    epoch_annotations[f'{event_type}_count'] += 1
                    epoch_annotations[f'has_{event_type}'] = 1
                    break
    return epoch_annotations

def extract_eeg_features_for_apnea(epoch_data: np.ndarray, fs: int,
                                    prefix: str) -> Dict[str, float]:
    features = {}
    try:
        freqs, psd = signal.welch(epoch_data, fs, nperseg=min(128, len(epoch_data)))
        bands = {'delta': (0.5,4), 'theta': (4,8), 'alpha': (8,13),
                 'beta': (13,30), 'gamma': (30,45)}
        total_power = 0
        band_powers = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                power = np.sum(psd[mask])
                band_powers[band_name] = power
                features[f'{prefix}{band_name}_power'] = float(power)
                total_power += power
        if total_power > 0:
            for band_name in bands:
                features[f'{prefix}{band_name}_ratio'] = float(band_powers[band_name] / total_power)
            slow = band_powers.get('delta',0)+band_powers.get('theta',0)
            fast = band_powers.get('alpha',0)+band_powers.get('beta',0)+1e-10
            features[f'{prefix}slow_fast_ratio'] = float(slow / fast)

        # энтропия
        data_norm = epoch_data - np.min(epoch_data)
        if np.sum(data_norm) > 0:
            prob = data_norm / np.sum(data_norm)
            features[f'{prefix}shannon_entropy'] = -np.sum(prob * np.log2(prob + 1e-10))

        # статистики
        mean_val = np.mean(epoch_data)
        std_val = np.std(epoch_data)
        features[f'{prefix}mean'] = float(mean_val)
        features[f'{prefix}std'] = float(std_val)
        features[f'{prefix}min'] = float(np.min(epoch_data))
        features[f'{prefix}max'] = float(np.max(epoch_data))
        features[f'{prefix}range'] = float(np.ptp(epoch_data))
        features[f'{prefix}rms'] = float(np.sqrt(np.mean(epoch_data**2)))
        if mean_val != 0:
            features[f'{prefix}cv'] = float(std_val / abs(mean_val))
        features[f'{prefix}skewness'] = float(skew(epoch_data))

        # динамические
        zc = np.sum(np.abs(np.diff(np.sign(epoch_data))) > 0)
        features[f'{prefix}zc_rate'] = float(zc / len(epoch_data))
        diff1 = np.diff(epoch_data)
        var_x = np.var(epoch_data)
        var_dx = np.var(diff1) if len(diff1)>0 else 0
        if var_x > 0:
            features[f'{prefix}mobility'] = float(np.sqrt(var_dx / var_x))
    except Exception:
        pass
    return features

def extract_epochs_eeg_only(raw: mne.io.Raw, annotations: mne.Annotations,
                            epoch_duration: int = 30):
    data, _ = raw[:, :]
    total_samples = data.shape[1]
    fs = int(raw.info['sfreq'])
    epoch_samples = int(epoch_duration * fs)

    eeg_channels = ['EEG F3-A2', 'EEG C3-A2', 'EEG O1-A2',
                    'EEG F4-A1', 'EEG C4-A1', 'EEG O2-A1']
    epochs_dict = {}
    annotations_list = []

    for ch_name in raw.ch_names:
        if ch_name not in eeg_channels:
            continue
        ch_idx = raw.ch_names.index(ch_name)
        channel_data = data[ch_idx]
        epochs = []
        start_sample = 0
        epoch_idx = 0
        while start_sample + epoch_samples <= total_samples:
            end_sample = start_sample + epoch_samples
            epoch = channel_data[start_sample:end_sample]
            epochs.append(epoch)
            if ch_name == eeg_channels[0]:
                epoch_anns = map_annotation_to_epoch_fixed(
                    start_sample/fs, end_sample/fs, annotations
                )
                epoch_anns['epoch_idx'] = epoch_idx
                epoch_anns['start_time'] = start_sample/fs
                epoch_anns['end_time'] = end_sample/fs
                annotations_list.append(epoch_anns)
            start_sample += epoch_samples
            epoch_idx += 1
        epochs_dict[ch_name] = np.array(epochs)
    return epochs_dict, annotations_list

def process_single_file(file_path: str, patient_id: int) -> Optional[pd.DataFrame]:
    """Обрабатывает один EDF-файл и возвращает DataFrame."""
    raw, annotations = load_psg_with_annotations(file_path)
    if raw is None:
        return None
    raw_processed = preprocess_eeg_only(raw)
    if raw_processed is None:
        return None
    epochs_dict, annotations_list = extract_epochs_eeg_only(raw_processed, annotations)
    if not epochs_dict:
        return None

    total_epochs = len(epochs_dict[list(epochs_dict.keys())[0]])
    fs = int(raw_processed.info['sfreq'])
    features_list = []
    for epoch_idx in range(total_epochs):
        epoch_features = {}
        for ch_name, epochs in epochs_dict.items():
            epoch_data = epochs[epoch_idx]
            prefix = f"{ch_name}_"
            eeg_features = extract_eeg_features_for_apnea(epoch_data, fs, prefix)
            epoch_features.update(eeg_features)
        if epoch_idx < len(annotations_list):
            epoch_features.update(annotations_list[epoch_idx])
        epoch_features['patient_id'] = patient_id
        features_list.append(epoch_features)
    return pd.DataFrame(features_list)

# -------------------------------------------------------------------
#  Параллельная обработка с кэшированием
# -------------------------------------------------------------------

def get_file_hash(filepath: str) -> str:
    """Быстрый хеш (размер + mtime) для проверки изменений."""
    stat = os.stat(filepath)
    return f"{stat.st_size}_{stat.st_mtime}"

def process_file_with_cache(file_path: str, patient_id: int, cache_dir: str) -> Optional[pd.DataFrame]:
    """Обрабатывает файл, сохраняя результат в кэш по имени файла+хеш."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(cache_dir, f"{base}_{file_hash}.pkl")
    if os.path.exists(cache_file):
        # Загружаем из кэша
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    # Обрабатываем
    df = process_single_file(file_path, patient_id)
    if df is not None:
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
    return df

def process_all_parallel(input_dir: str, output_csv: str, cache_dir: str = "cache",
                         n_workers: int = 8):
    """
    Параллельная обработка всех EDF файлов.
    n_workers – количество одновременных процессов.
    """
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.edf')]
    if not all_files:
        print("❌ Нет EDF файлов в указанной папке.")
        return

    full_paths = [os.path.join(input_dir, f) for f in all_files]
    print(f"🔍 Найдено файлов: {len(full_paths)}")
    print(f"🚀 Запуск с {n_workers} процессами...")

    # Функция-обёртка для передачи дополнительных аргументов
    worker_func = partial(process_file_with_cache, cache_dir=cache_dir)

    results = []
    # Используем multiprocessing.Pool с tqdm для прогресса
    from multiprocessing import Pool, cpu_count
    # Ограничим разумным числом (не больше числа файлов и не больше 2*CPU)
    n_workers = min(n_workers, len(full_paths), cpu_count()*2)
    with Pool(processes=n_workers) as pool:
        # starmap? нет, у нас одна переменная + id
        # передаём (file_path, patient_id)
        tasks = [(path, idx+1) for idx, path in enumerate(full_paths)]
        for df in tqdm(pool.starmap(worker_func, tasks), total=len(tasks), desc="Обработка файлов"):
            if df is not None:
                results.append(df)

    if not results:
        print("❌ Не удалось обработать ни одного файла.")
        return

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ Сохранено {len(final_df)} эпох в {output_csv}")

# -------------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Параллельное извлечение признаков ЭЭГ из EDF")
    parser.add_argument("--input_dir", "-i", default=r"F:\MedDB\psg\edf", help="Папка с EDF файлами")
    parser.add_argument("--output", "-o", default="eeg_dataset.csv", help="Выходной CSV")
    parser.add_argument("--cache_dir", "-c", default="cache", help="Папка для кэша (промежуточных результатов)")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Количество параллельных процессов (по умолчанию 10)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: папка '{args.input_dir}' не найдена.")
        return

    process_all_parallel(args.input_dir, args.output, args.cache_dir, n_workers=args.workers)

if __name__ == "__main__":
    main()