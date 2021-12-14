from c_song import Song
from utils import *
from laplacian_utils import get_split_times
from utils_plot import generate_plots, compare_mel_arrays_between_layers
from utils_midi import save_peak_notes, generate_sin_waveform, save_midi_from_note_packets, save_midi_from_note_rows, get_dominant_notes_from_note_rows

# =====================================================================================================================
# + ACTION!
# =====================================================================================================================
song_name = "beethoven"
song_path = "songs/{}/{}.flac".format(song_name, song_name)
convert(song_path, "wav")
wav_path = song_path.replace(".flac", ".wav")
split_times = get_split_times(wav_path)
song = Song(song_path, split_times)
save_click_track_wavs(song=song)

# layer = "other"
# generate_plots(song=song)
# compare_mel_arrays_between_layers(song=song, layer_a=layer, layer_b="original")
# generate_sin_waveform(song=song, layer=layer)
# save_peak_notes(song=song, layer=layer)
# save_midi_from_note_packets(song=song, layer=layer)
# save_midi_from_note_rows(song=song, layer=layer)
# get_dominant_notes_from_note_rows(song=song, layer=layer)