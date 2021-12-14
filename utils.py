from pydub import AudioSegment
import soundfile as sf
import os

def save_click_track_wavs(song):
    clicktracks_dir_path = song.dir_paths.get("clicktracks")
    if not len(os.listdir(clicktracks_dir_path)) == 0:
        print("Click tracks folder not empty. Aborting...")
        return

    for layer_name, layer_splits in song.layers.items():
        clicktracks_layer_dir_path = "{}/{}".format(clicktracks_dir_path, layer_name)
        os.mkdir(clicktracks_layer_dir_path)
        for split in layer_splits:
            split_basename = split.split('.')[0]
            click_track_path = '{}/{}_click.wav'.format(clicktracks_layer_dir_path, split_basename)
            sf.write(click_track_path, split.np_clicks_y, samplerate=split.sr)

def convert(song_path, target_format):
    original_format = song_path.split('.')[-1]
    target_path = song_path.replace(".{}".format(original_format), ".{}".format(target_format))
    if os.path.exists(target_path):
        print("{} already exists.".format(target_format))
        return

    audio = AudioSegment.from_file(song_path)
    audio.set_channels(1)
    audio.export(target_path, target_format)

def file_exists(dir_name, file_name):
    peak_notes_file_path = "{}/{}".format(dir_name, file_name)
    if os.path.exists(peak_notes_file_path):
        return True
    return False