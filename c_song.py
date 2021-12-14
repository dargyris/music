from c_layer import Layer
import os
import librosa
from pydub import AudioSegment
import pickle

class Song:
    def __init__(self, path, split_times):
        self.path = path
        self.split_times = split_times
        self.name = self.make_name()
        self.dir_paths = self.make_dir_paths()

        self.layers = self.load_layer_data()
        if self.layers is None:
            self.create_dir_tree()
            self.create_song_layers_with_spleet()
            self.create_layer_splits()
            self.layers = self.populate_layers_dict()
            self.save_data()

    def make_name(self):
        song_name_with_extension = os.path.basename(self.path)
        return song_name_with_extension.split('.')[0]

    def load_layer_data(self):
        song_data_path = "{}/{}{}".format(self.dir_paths.get("data"), self.name, ".pickle")
        if not os.path.exists(song_data_path):
            return None

        with open(song_data_path, 'rb') as f:
            layers = pickle.load(f)
            f.close()
        return layers

    # + CREATE DIRS
    def make_dir_paths(self):
        dir_names = ["spleet", "splits", "sine", "plots", "midi", "clicktracks", "peak", "data"]
        dir_paths = {"root": os.path.dirname(self.path)}
        for dir_name in dir_names:
            dir_paths[dir_name] = "{}/{}".format(dir_paths.get("root"), dir_name)
        return dir_paths

    def create_dir_if_not_exists(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def create_dir_tree(self):
        [self.create_dir_if_not_exists(path) for path in self.dir_paths.values() if "spleet" not in path]

    # + CREATE LAYERS WITH SPLEET
    def spleet_song(self):
        spleet_dir_path = self.dir_paths.get("root") + '/{}'.format(self.name)
        os.system("spleeter separate -p spleeter:5stems -o {} {}".format(self.dir_paths.get("root"), self.path))
        os.rename(spleet_dir_path, self.dir_paths.get("spleet"))

    def copy_layer_original_wav_in_spleet_dir(self):
        path_for_original_wav_in_spleet_dir = self.dir_paths.get("spleet") + "/original.wav"
        os.system("cp {} {}".format(self.path, path_for_original_wav_in_spleet_dir))

    def create_layer_splits(self):
        split_dir_path = self.dir_paths.get("splits")
        spleet_dir_path = self.dir_paths.get("spleet")
        for file in os.listdir(spleet_dir_path):
            layer_name = file.split('.')[0]
            split_layer_dir_path = "{}/{}".format(split_dir_path, layer_name)
            os.mkdir(split_layer_dir_path)
            song_path = "{}/{}".format(spleet_dir_path, file)
            split_times = self.split_times
            for t_index in range(len(split_times) - 1):
                from_t_millis = split_times[t_index] * 1000
                to_t_millis = split_times[t_index + 1] * 1000
                part = AudioSegment.from_file(song_path)
                part = part[from_t_millis:to_t_millis]
                part.export("{}/{}.wav".format(split_layer_dir_path, t_index), format="wav")

    def reduce_all_layers_to_single_channel(self):
        spleet_dir_path = self.dir_paths.get("spleet")
        for file_name in os.listdir(spleet_dir_path):
            file_path = spleet_dir_path + '/{}'.format(file_name)
            sound = AudioSegment.from_file(file_path)
            sound = sound.set_channels(1)
            sound.export(file_path, format="wav")

    def create_song_layers_with_spleet(self):
        if os.path.exists(self.dir_paths.get("spleet")):
            print("Song already split...")
            return

        self.spleet_song()
        self.copy_layer_original_wav_in_spleet_dir()
        self.reduce_all_layers_to_single_channel()

    # + POPULATE DICT AND SAVE DATA
    def populate_layers_dict(self):
        splits_dir_path = self.dir_paths.get("splits")
        layers = {}
        for layer_dir in os.listdir(splits_dir_path):
            layer_name = layer_dir.split('/')[-1]
            layer_dir_path = "{}/{}".format(splits_dir_path, layer_dir)
            layer_splits = []
            for layer_split_wav in os.listdir(layer_dir_path):
                file_path = '{}/{}'.format(layer_dir_path, layer_split_wav)
                y, sr = librosa.load(file_path)
                layer_splits.append(Layer(y=y, sr=sr))
            layers[layer_name] = layer_splits
        return layers

    def save_data(self):
        data_file_path = '{}/{}{}'.format(self.dir_paths.get("data"), self.name, ".pickle")
        with open(data_file_path, 'wb+') as f:
            pickle.dump(self.layers, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
