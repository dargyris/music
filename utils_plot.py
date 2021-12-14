import os
import matplotlib.pyplot as plt
import librosa
from librosa import display
import numpy as np

def save_subplots(song, type):
    func_action_msg = "Generating subplots for {} (all layers)".format(type)
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}{}".format(song.dir_paths.get("plots"), type, "_subplots.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(30, 25), dpi=600)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6,1,subplot_counter)
        plt.title(layer_name)
        librosa.display.waveplot(y=layer_data.np_y, sr=layer_data.sr)
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def save_subplots_overlaid_waveforms(song):
    func_action_msg = "Generating overlay plots for all wav files (waveform vs click track)"
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}".format(song.dir_paths.get("plots"), "overlaid_subplots.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(30, 25), dpi=600)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6, 1, subplot_counter)
        plt.title(layer_name)
        librosa.display.waveplot(y=layer_data.np_y, sr=layer_data.sr)
        librosa.display.waveplot(y=layer_data.np_clicks_y, sr=layer_data.sr)
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def save_mel_spectrograms(song):
    func_action_msg = "Generating mel spectrograms"
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}".format(song.dir_paths.get("plots"), "mel_spectrograms.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(30, 25), dpi=600)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6,1,subplot_counter)
        plt.title(layer_name)
        spectrogram = layer_data.np_mels
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        librosa.display.specshow(spectrogram_db, sr=layer_data.sr, fmax=20000)
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def save_yin_mel_overlay(song):
    func_action_msg = "Generating yin - mel overalay plots"
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}".format(song.dir_paths.get("plots"), "pyin_mel.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(25, 20), dpi=400)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6,1,subplot_counter)
        plt.title(layer_name)
        D = layer_data.D
        librosa.display.specshow(D, x_axis="time", fmax=20000)
        f0 = layer_data.pyin_f0_vf_vp[0]
        times = librosa.times_like(f0, sr=layer_data.sr)
        plt.plot(times, f0, color="cyan", linewidth=1)
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def save_power_log_overlay(song):
    func_action_msg = "Generating power log overlay"
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}".format(song.dir_paths.get("plots"), "pwr_log_overlay.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(25, 20), dpi=400)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6, 1, subplot_counter)
        plt.title(layer_name)
        S = layer_data.S_phase[0]
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time')
        cent = librosa.feature.spectral_centroid(y=layer_data.np_y, sr=layer_data.sr)
        times = librosa.times_like(cent)
        plt.plot(times, cent.T, label='Spectral centroid', color='w')
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def save_waveform_pitch_overlay(song):
    func_action_msg = "Generating overlay plots: pitch on mel spectrogram for all layers"
    print("\t> {}...".format(func_action_msg))

    plot_path = "{}/{}".format(song.dir_paths.get("plots"), "overlaid_pitch.png")
    if os.path.exists(plot_path):
        print("Plot already exists!")
        return

    plt.figure(figsize=(25, 20), dpi=400)
    subplot_counter = 1
    for layer_name, layer_data in song.layers.items():
        plt.subplot(6, 1, subplot_counter)
        plt.title(layer_name)
        S = layer_data.S_phase[0]
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time')
        freq = layer_data.crepe_tfca[1]
        times = librosa.times_like(freq)
        plt.plot(times, freq, label='Spectral centroid', color='w')
        subplot_counter += 1
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

def generate_plots(song):
    save_subplots(song, "data")
    save_subplots(song, "clicks_data")

    save_subplots_overlaid_waveforms(song)
    save_mel_spectrograms(song)
    save_yin_mel_overlay(song)
    save_power_log_overlay(song)
    save_waveform_pitch_overlay(song)

def print_mel_array_header(spectrogram, from_t_window, to_t_window):
    print("Max: {:.2f}\tMin: {:.2f}\t Shape: {}\tmels (rows): {}\tt windows (columns): {}\tFrom: {}\tTo: {}".format(
        np.max(spectrogram),
        np.min(spectrogram),
        spectrogram.shape,
        spectrogram.shape[0],
        spectrogram.shape[1],
        from_t_window,
        to_t_window - 1)
    )

# + MEL ARRAY COMPARISON BETWEEN LAYERS (_T)
def print_mel_array_body(spectrogram, from_t_window, to_t_window):
    for column_index in range(from_t_window, to_t_window):
        column = spectrogram[:, column_index]
        column_conditional = column > np.mean(spectrogram[:, column_index])
        column_list = column_conditional.tolist()
        pretty_list = ['T' if elem else '_' for elem in column_list]
        print(column_index, end=' [')
        [print(elem, end='') for elem in pretty_list]
        print(']')

def print_mel_array(song, layer, from_t_window, to_t_window):
    print("\n> MEL _T ARRAYS - Transposed View\tSong: {}".format(song.name))

    np_mels = song.layers.get(layer).np_mels
    print_mel_array_header(np_mels, from_t_window, to_t_window)
    print_mel_array_body(np_mels, from_t_window, to_t_window)
    print('\n')

def compare_mel_arrays_between_layers(song, layer_a, layer_b):
    print_mel_array(song, layer=layer_a, from_t_window=50, to_t_window=100)
    print_mel_array(song, layer=layer_b, from_t_window=50, to_t_window=100)
