import matplotlib.pyplot
import numpy as np
import librosa
import os
from scipy.io import wavfile
import pydsm
from utils import file_exists
import pretty_midi

N_MELS = 256
FMAX = 8000.

def make_adjustment_factors(from_idx, to_idx, c, s, p):
    arr_sigm = np.linspace(from_idx, to_idx, num=N_MELS)
    arr_sigm = 1 / (1 + np.exp(-c * (arr_sigm))) + np.ones(N_MELS)
    arr_sigm /= np.max(arr_sigm)
    arr_sigm = arr_sigm ** p
    arr_sigm += np.ones(N_MELS) * s
    return arr_sigm

def make_loudness_table_from_pydsm_iso():
    t_freq_loudness = pydsm.iso226.iso226_spl_contour(L_N=40, hfe=False)
    loudness_thresholds = t_freq_loudness[1][:-2]
    inverse_loudness = 1 / loudness_thresholds
    inverse_loudness /= np.max(inverse_loudness)
    element_repeat_factor = int(np.ceil(N_MELS / len(inverse_loudness)))
    inverse_loudness = np.repeat(inverse_loudness, element_repeat_factor)
    inverse_loudness = inverse_loudness[:N_MELS]
    return inverse_loudness

def make_loudness_adjustment_threshold_table():
    adjustment_factors = make_adjustment_factors(from_idx=-20, to_idx=80, c=0.05, s=2, p=2)
    pydsm_iso_loudness_table = make_loudness_table_from_pydsm_iso()
    adj_inverse_loudness = pydsm_iso_loudness_table * adjustment_factors
    return adj_inverse_loudness

def adjust_spectrogram_with_loudness_contour(np_mels):
    loudness_thresholds = make_loudness_adjustment_threshold_table()
    num_twindows = len(np_mels[0])
    for twindow_index in range(num_twindows):
        column = np_mels[:, twindow_index]
        column *= loudness_thresholds
        np_mels[:, twindow_index] = column
    return np_mels

def cutoff_mels_below_col_mean(np_mels):
    num_melframes = len(np_mels[0])
    filtered_np_mels = np.ndarray(np_mels.shape)
    for frame_index in range(num_melframes):
        mel_frame = np_mels[:, frame_index]
        mel_frame_mean = np.mean(mel_frame)
        hz_frame = [elem if elem > mel_frame_mean else 0 for elem in mel_frame]
        filtered_np_mels[:, frame_index] = hz_frame
    return filtered_np_mels

def make_notes_template():
    mel_hz_centers = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX, htk=False)
    mel_hz_centers[0] = 1e-30
    freq_to_note_table = librosa.hz_to_note(mel_hz_centers)
    freq_to_note_table[0] = '___'
    return freq_to_note_table

def turn_active_mels_to_notes(np_mels, notes_template):
    num_frames = len(np_mels[0])
    note_peaks_table = np.ndarray(np_mels.shape, dtype="object")
    for frame_index in range(num_frames):
        mel_column = np_mels[:, frame_index]
        note_column = ["{:<3}".format(note) if col_element > 0. else 3*"_" for note, col_element in zip(notes_template, mel_column)]
        note_peaks_table[:, frame_index] = note_column
    return note_peaks_table

def turn_active_mels_to_notes_with_loudness(np_mels, notes_template):
    num_frames = len(np_mels[0])
    max = np.max(np_mels)
    note_peaks_with_loudness_table = np.ndarray(np_mels.shape, dtype="object")
    for frame_index in range(num_frames):
        mel_frame = np_mels[:, frame_index]
        note_loudness_frame = ["|({:<3}),({:<.6f})|".format(note, loudness/max) if loudness > 0. else "_" for note, loudness in zip(notes_template, mel_frame)]
        note_peaks_with_loudness_table[:, frame_index] = note_loudness_frame
    return note_peaks_with_loudness_table

def make_printable_notes(table, song_duration):
    notes_lines = []
    num_frames = len(table[0])
    frame_duration = song_duration / num_frames
    for frame_index in range(num_frames):
        frame = (table[:, frame_index]).tolist()
        frame_timestamp = float(frame_index * frame_duration)
        row_to_write = "{:<3} {:<4.2f} [ ".format(frame_index, frame_timestamp)
        for elem in frame:
            row_to_write += "{}".format(elem, end=' ')
        row_to_write += " ]\n"
        notes_lines.append(row_to_write)
    return notes_lines

def save_to_file(song, layer, table):
    pmels_dir_path = song.dir_paths.get("peak")
    notes_file_path = "{}/{}_notes.txt".format(pmels_dir_path, layer)
    with open(notes_file_path, 'w+') as f:
        f.writelines(table)
        f.close()

def save_peak_notes_to_txt(song, filtered_np_mels, layer):
    notes_template = make_notes_template()
    notes_peaks_table = turn_active_mels_to_notes(filtered_np_mels, notes_template)
    notes_peaks_with_loudness_table = turn_active_mels_to_notes_with_loudness(filtered_np_mels, notes_template)

    notes_printable = make_printable_notes(table=notes_peaks_table, song_duration=song.layers.get(layer).t_duration_sec_hms[0])
    notes_with_loudness_printable = make_printable_notes(table=notes_peaks_with_loudness_table, song_duration=song.layers.get(layer).t_duration_sec_hms[0])

    save_to_file(song=song, layer=layer, table=notes_printable)
    save_to_file(song=song, layer=layer+'_ld', table=notes_with_loudness_printable)

def save_peak_notes(song, layer):
    func_action_msg = "Creating peak notes txt file"
    print("\t> {}...".format(func_action_msg))

    if file_exists(dir_name=song.dir_paths.get("peak"), file_name="{}_notes.txt".format(layer)):
        print("Peak notes files already exist!")
        return

    np_mels = song.layers.get(layer).np_mels
    np_mels = librosa.power_to_db(np_mels)
    adj_np_mels = adjust_spectrogram_with_loudness_contour(np_mels)
    filtered_np_mels = cutoff_mels_below_col_mean(adj_np_mels)
    save_peak_notes_to_txt(song, filtered_np_mels, layer)

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

# =====================================================================================================================
# + GENERATE SIMPLE SIN WAVEFORM:
# =====================================================================================================================
def make_hz_template():
    hz_template = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX, htk=False)
    hz_template[0] = 1e-30
    hz_template = np.round(hz_template)
    hz_template = [np.int(elem) for elem in hz_template]
    return hz_template

# def make_hz_peaks_table(adj_np_mels, hz_template):
#     num_twindows = len(adj_np_mels[0])
#     hz_peaks_table = np.ndarray(adj_np_mels.shape)
#     for twindow_index in range(num_twindows):
#         column = adj_np_mels[:, twindow_index]
#         column = [0 if elem == 0 else 1 for elem in column]
#         hz_column = column * hz_template
#         hz_peaks_table[:, twindow_index] = hz_column
#     return hz_peaks_table

def make_constant_loudness_sin_ys(song_duration, total_samples):
    hz_template = make_hz_template()
    y_rows = []
    for row_freq in hz_template:
        np_col_duration = np.linspace(0, song_duration, total_samples)
        y_row = np.sin(row_freq * 2 * np.pi * np_col_duration)
        y_rows.append(y_row)
    return np.array(y_rows)

def make_loudness_rows_samples(layer_data, total_samples):
    np_mels = layer_data.np_mels
    adj_np_mels = adjust_spectrogram_with_loudness_contour(np_mels)
    filtered_np_mels = cutoff_mels_below_col_mean(adj_np_mels)
    loudness_rows = []
    num_mel_columns = len(np_mels[0])
    element_repeat_factor = int(np.ceil(total_samples / num_mel_columns))
    for row in filtered_np_mels:
        row = np.repeat(row, element_repeat_factor)
        row = row[:total_samples]
        loudness_rows.append(row)
    return np.array(loudness_rows)

def save_ys_for_all_mel_frequencies(y_sin_rows_constant_loudness, song, layer, sr):
    for i in range(len(y_sin_rows_constant_loudness)):
        y = y_sin_rows_constant_loudness[i]
        sine_file_path = "{}/{}{}{}".format(song.dir_paths.get("sine"), layer, i, "_part.wav")
        wavfile.write(sine_file_path, sr, y)

def merge_saved_ys_to_one_wav_file_and_save(sine_dir_path, layer):
    sox_argument = ' '.join(["{}/{}".format(sine_dir_path, file) for file in os.listdir(sine_dir_path)])
    sox_argument += " {}/{}".format(sine_dir_path, "{}_sine.wav".format(layer))
    os.system("sox -m {}".format(sox_argument))

def delete_wavs_for_single_frequencies(sine_dir_path):
    rm_argument = ' '.join(["{}/{}".format(sine_dir_path, file) for file in os.listdir(sine_dir_path) if not "sine" in file])
    os.system("rm -rf {}".format(rm_argument))

def generate_sin_waveform(song, layer):
    func_action_msg = "Generating sin waveforms for all mels, Layer: {}".format(layer)
    print("\t> {}...".format(func_action_msg))

    if file_exists(dir_name=song.dir_paths.get("sine"), file_name="{}_sine.wav".format(layer)):
        print("Sine.wav already exists!")
        return

    layer_data = song.layers.get(layer)
    sr = layer_data.sr
    song_duration = layer_data.t_duration_sec_hms[0]
    total_samples = int(sr * song_duration)

    y_sin_rows_constant_loudness = make_constant_loudness_sin_ys(song_duration, total_samples)
    np_loudness_rows = make_loudness_rows_samples(layer_data, total_samples)
    y_sin_rows = y_sin_rows_constant_loudness * np_loudness_rows

    save_ys_for_all_mel_frequencies(y_sin_rows, song, layer, sr)
    merge_saved_ys_to_one_wav_file_and_save(sine_dir_path=song.dir_paths.get("sine"), layer=layer)
    delete_wavs_for_single_frequencies(song.dir_paths.get("sine"))

    print("\t> {}".format(func_action_msg), end="\t\t[DONE]\n")

# =====================================================================================================================
# + GENERATE MIDI FILE
# + ===================================================================================================================
def make_loudness_rows(layer_data):
    # np_mels = librosa.power_to_db(layer_data.np_mels)
    np_mels = librosa.amplitude_to_db(np.abs(layer_data.np_mels))
    np_mels = np_mels.clip(min=0)
    adj_np_mels = adjust_spectrogram_with_loudness_contour(np_mels)
    filtered_np_mels = cutoff_mels_below_col_mean(adj_np_mels)
    filtered_np_mels /= np.max(filtered_np_mels)
    filtered_np_mels *= 127
    return filtered_np_mels.tolist()

def create_note_rows(song, layer):
    mel_hz_centers = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX, htk=False)
    layer_data = song.layers.get(layer)
    loudness_rows = make_loudness_rows(layer_data=layer_data)
    note_rows = [(note_hz, loudness_row) for note_hz, loudness_row in zip(mel_hz_centers, loudness_rows[:])]
    return note_rows
# + -------------------------------------------------------------------------------------------------------------------
def plot_note_rows_loudness(note_rows):
    import matplotlib.pyplot as plt
    non_problematic_rows = note_rows[2:]
    for row in non_problematic_rows:
        note_hz = row[0]
        note_number = int(pretty_midi.hz_to_note_number(note_hz))
        note_name = pretty_midi.note_number_to_name(note_number)
        plt.plot(row[1], label=note_name)
        plt.legend()
        plt.title(note_name)
    plt.show()
# + -------------------------------------------------------------------------------------------------------------------
def make_rows_with_loudness_packets(song, layer, note_rows):
    rows_with_loudness_packets = []
    for row in note_rows:
        loudnesses = row[1]

        packets = []
        new_packet = {
            'is_active': False,
            'start': 0,
            'end': 0,
            'velocities': []
        }
        num_frames = len(loudnesses)
        song_duration = song.layers.get(layer).t_duration_sec_hms[0]
        frame_duration = song_duration / num_frames
        for frame_index in range(num_frames):
            velocity = int(loudnesses[frame_index])
            if velocity <= 10:
                if len(packets) == 0:
                    continue
                current_packet = packets[-1]
                if current_packet.get("is_active"):
                    current_packet["is_active"] = False
                    current_packet["velocities"].append(velocity)
                    current_packet["end"] = frame_duration * (frame_index + 1)
            else:
                if len(packets) == 0:
                    new_packet["is_active"] = True
                    new_packet["start"] = frame_duration * frame_index
                    new_packet["velocities"].append(velocity)
                    packets.append(new_packet)
                    continue
                current_packet = packets[-1]
                if current_packet["is_active"]:
                    current_packet["velocities"].append(velocity)
                else:
                    new_packet["is_active"] = True
                    new_packet["start"] = frame_duration * frame_index
                    new_packet["velocities"].append(velocity)
                    packets.append(new_packet)

        note_hz = row[0] if row[0] > 5 else 10
        note_number = int(pretty_midi.hz_to_note_number(note_hz))
        rows_with_loudness_packets.append((note_number, packets))
    # [print(list(row)) for row in enumerate(rows_with_loudness_packets)]
    # print(len(rows_with_loudness_packets))
    # input()
    return rows_with_loudness_packets

# + -------------------------------------------------------------------------------------------------------------------
def save_midi_from_note_packets(song, layer):
    note_rows = create_note_rows(song, layer)
    # plot_note_rows_loudness(note_rows)
    note_rows_packets = make_rows_with_loudness_packets(song, layer, note_rows)

    midi = pretty_midi.PrettyMIDI()
    midi_instr = pretty_midi.instrument_name_to_program('Cello')
    instr_midi = pretty_midi.Instrument(program=midi_instr)

    for row in note_rows_packets:
        note_number = row[0]
        note_packets = row[1]
        for packet in note_packets:
            start = packet.get("start")
            end = packet.get("end")
            velocity = int(np.mean(packet.get("velocities")))
            note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=start, end=end)
            instr_midi.notes.append(note)
    midi.instruments.append(instr_midi)

    midi_dir_path = song.dir_paths.get("midi")
    midi_file_name = "{}_packets.mid".format(layer)
    file_path = "{}/{}".format(midi_dir_path, midi_file_name)
    midi.write(file_path)

def save_midi_from_note_rows(song, layer):
    note_rows = create_note_rows(song, layer)

    num_frames = len(note_rows[0][1])
    song_duration = song.layers.get(layer).t_duration_sec_hms[0]
    frame_duration = song_duration / num_frames

    midi = pretty_midi.PrettyMIDI()
    midi_instr = pretty_midi.instrument_name_to_program('Cello')
    instr_midi = pretty_midi.Instrument(program=midi_instr)

    for note_row in note_rows:
        note_hz = note_row[0] if note_row[0] > 5 else 10
        note_number = int(pretty_midi.hz_to_note_number(note_hz))
        for frame_index in range(num_frames):
            start = frame_duration * frame_index
            end = frame_duration * (frame_index + 1)
            velocities = note_row[1]
            velocity = int(velocities[frame_index])
            note = pretty_midi.Note(velocity=velocity, pitch=note_number, start=start, end=end)
            instr_midi.notes.append(note)
    midi.instruments.append(instr_midi)

    midi_dir_path = song.dir_paths.get("midi")
    midi_file_name = "{}_rows.mid".format(layer)
    file_path = "{}/{}".format(midi_dir_path, midi_file_name)
    midi.write(file_path)

def get_dominant_notes_from_note_rows(song, layer):
    note_rows = create_note_rows(song, layer)
    note_freq_of_appearances = {}
    for note_row in note_rows:
        volumes = note_row[1]
        freq_of_appearance = []
        splits = 3
        max_volume = 127
        low_threshold = 10
        for weight in range(splits):
            vol_threshold = weight*(max_volume/splits) + low_threshold
            freq_of_appearance.append(len([vol for vol in volumes if vol > vol_threshold]) * (weight+1))

        note_hz = note_row[0] if note_row[0] > 5 else 10
        note_number = int(pretty_midi.hz_to_note_number(note_hz))
        note_name = pretty_midi.note_number_to_name(note_number)
        if note_name in note_freq_of_appearances.keys():
            note_freq_of_appearances.get(note_name)['sum_foa'] += sum(freq_of_appearance)
            note_freq_of_appearances.get(note_name)['num_occurrances'] += 1
        else:
            note_freq_of_appearances[note_name] = {'sum_foa': sum(freq_of_appearance), 'num_occurrances': 1}

    note_freq_of_appearances_list = [(k, v) for k, v in note_freq_of_appearances.items()]
    note_freq_of_appearances_list = sorted(note_freq_of_appearances_list, key=lambda x: x[1].get("sum_foa"), reverse=True)
    # [print(list(row)) for row in note_freq_of_appearances_list]

    note_freq_of_appearances_list = sorted(note_freq_of_appearances_list, key=lambda x: x[0][:-1])
    summed_foas = {}
    for row_index in range(len(note_freq_of_appearances_list)):
        row = note_freq_of_appearances_list[row_index]
        note_letters = row[0][:-1]
        row_sum_foa = row[1].get("sum_foa")
        if note_letters in summed_foas.keys():
            summed_foas.get(note_letters)["row_sum_foa"] += row_sum_foa
        else:
            summed_foas[note_letters] = {"row_sum_foa": row_sum_foa}

    summed_foas_list = [(k,v) for k, v in summed_foas.items()]
    summed_foas_list = sorted(summed_foas_list, key=lambda x: x[1].get('row_sum_foa'), reverse=True)
    [print(list(row)) for row in summed_foas_list]
    input()
