import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 1000000
matplotlib.use('TkAgg')
import crepe

N_MELS = 256
FMAX = 8000.

class Layer:
    def __init__(self, y, sr):
        self.np_y = y
        self.sr = sr

        self.t_duration_sec_hms = self.make_duration()
        self.t_rhythm_tempo_beats = self.make_rhythm()
        self.np_clicks_y = self.make_clicks_waveform()
        self.pyin_f0_vf_vp = self.make_pyin_values()
        self.crepe_tfca = self.make_crepe_values()
        self.np_mels = self.make_mels()
        self.np_stft = self.make_stft()
        self.S_phase = self.make_S_phase()
        self.amplitude = self.make_amplitude()
        self.D = self.make_D()

    def make_duration(self):
        duration_seconds = librosa.get_duration(y=self.np_y, sr=self.sr)
        d = duration_seconds
        hms_duration = str(int(d // 3600)) + 'h\t'
        d %= 3600
        hms_duration += str(int(d // 60)) + 'm\t'
        d %= 60
        hms_duration += str(int(d)) + 's'
        return duration_seconds, hms_duration

    def make_rhythm(self):
        tempo, beats = librosa.beat.beat_track(y=self.np_y, sr=self.sr)
        beats = [0] if len(beats) == 0 else beats
        return tempo, beats

    def make_clicks_waveform(self):
        beats = self.t_rhythm_tempo_beats[1]
        return librosa.clicks(frames=beats, sr=self.sr)

    def make_pyin_values(self):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=self.np_y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        return f0, voiced_flag, voiced_probs

    def make_crepe_values(self):
        times, frequencies, confidences, activations = crepe.predict(audio=self.np_y, sr=self.sr, viterbi=True)
        return times, frequencies, confidences, activations

    def make_mels(self):
        mels = librosa.feature.melspectrogram(y=self.np_y, sr=self.sr, n_mels=N_MELS, fmax=FMAX)
        return mels

    def make_stft(self):
        stft = librosa.stft(y=self.np_y)
        return stft

    def make_S_phase(self):
        S, phase = librosa.magphase(self.np_stft)
        return S, phase

    def make_amplitude(self):
        amplitude = np.abs(self.np_stft)
        return amplitude

    def make_D(self):
        D = librosa.amplitude_to_db(self.amplitude, ref=np.max)
        return D