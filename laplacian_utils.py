import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.cluster
import librosa
import librosa.display
import matplotlib.patches as patches
import os

def plot_cqt(wav_path, Csync, beat_times):
    fig, ax = plt.subplots()
    librosa.display.specshow(Csync, bins_per_octave=12*3, y_axis='cqt_hz', x_axis='time', x_coords=beat_times, ax=ax)
    png_path = "{}/{}.png".format(os.path.dirname(wav_path), "cvf_plots/cqt")
    plt.savefig(png_path)

def plot_recurrence_path_combo(wav_path, Rf, beat_times, R_path, A):
    fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(10, 4))
    librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[0])
    ax[0].set(title='Recurrence similarity')
    ax[0].label_outer()
    librosa.display.specshow(R_path, cmap='inferno_r', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
    ax[1].set(title='Path similarity')
    ax[1].label_outer()
    librosa.display.specshow(A, cmap='inferno_r', y_axis='time', x_axis='s', y_coords=beat_times, x_coords=beat_times, ax=ax[2])
    ax[2].set(title='Combined graph')
    ax[2].label_outer()
    png_path = "{}/{}.png".format(os.path.dirname(wav_path), "cvf_plots/recurrence_path_combo")
    plt.savefig(png_path)

def plot_recurrence_structure(wav_path, Rf, beat_times, X):
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', x_axis='time', y_coords=beat_times, x_coords=beat_times, ax=ax[1])
    ax[1].set(title='Recurrence similarity')
    ax[1].label_outer()
    librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
    ax[0].set(title='Structure components')
    png_path = "{}/{}.png".format(os.path.dirname(wav_path), "cvf_plots/recurrence_structure")
    plt.savefig(png_path)

def plot_recurrence_segments(wav_path, k, X, Rf, beat_times, seg_ids):
    fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 4))
    colors = plt.get_cmap('Paired', k)
    librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', y_coords=beat_times, ax=ax[1])
    ax[1].set(title='Recurrence matrix')
    ax[1].label_outer()
    librosa.display.specshow(X, y_axis='time', y_coords=beat_times, ax=ax[0])
    ax[0].set(title='Structure components')
    img = librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors, y_axis='time', y_coords=beat_times, ax=ax[2])
    ax[2].set(title='Estimated segments')
    ax[2].label_outer()
    fig.colorbar(img, ax=[ax[2]], ticks=range(k))
    png_path = "{}/{}.png".format(os.path.dirname(wav_path), "cvf_plots/recurrence_segments")
    plt.savefig(png_path)

def plot_time_split(wav_path, C, sr, BINS_PER_OCTAVE, k, bound_times, bound_segs, freqs):
    colors = plt.get_cmap('Paired', k)
    fig, ax = plt.subplots()
    librosa.display.specshow(C, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', ax=ax)
    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        ax.add_patch(patches.Rectangle((interval[0], freqs[0]), interval[1] - interval[0], freqs[-1], facecolor=colors(label), alpha=0.50))
    png_path = "{}/{}.png".format(os.path.dirname(wav_path), "cvf_plots/time_split")
    plt.savefig(png_path)

def get_split_times(wav_path):
    cvf_plots_dir_path = "{}/{}".format(os.path.dirname(wav_path), "cvf_plots")
    if os.path.exists(cvf_plots_dir_path):
        print("Cvf plots exist!")
        return

    os.mkdir(cvf_plots_dir_path)

    y, sr = librosa.load(wav_path)
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)
    beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0, x_max=C.shape[1]), sr=sr)
    plot_cqt(wav_path, Csync, beat_times)

    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
    A = mu * Rf + (1 - mu) * R_path
    plot_recurrence_path_combo(wav_path, Rf, beat_times, R_path, A)

    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    Cnorm = np.cumsum(evecs**2, axis=1)**0.5
    k = 5
    X = evecs[:, :k] / Cnorm[:, k-1:k]
    plot_recurrence_structure(wav_path, Rf, beat_times, X)

    KM = sklearn.cluster.KMeans(n_clusters=k)
    seg_ids = KM.fit_predict(X)
    plot_recurrence_segments(wav_path, k, X, Rf, beat_times, seg_ids)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
    bound_segs = list(seg_ids[bound_beats])
    bound_frames = beats[bound_beats]
    bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1]-1)
    bound_times = librosa.frames_to_time(bound_frames)
    freqs = librosa.cqt_frequencies(n_bins=C.shape[0], fmin=librosa.note_to_hz('C1'), bins_per_octave=BINS_PER_OCTAVE)
    plot_time_split(wav_path, C, sr, BINS_PER_OCTAVE, k, bound_times, bound_segs, freqs)

    bound_times_list = bound_times.tolist()
    time_lines = ["{:>3} :: {:>3}\n".format(int(elem/60), int(elem%60)) for elem in bound_times_list]
    txt_path = "{}/{}.txt".format(os.path.dirname(wav_path), "cvf_plots/time_splits")
    with open(txt_path, "w+") as f:
        f.writelines(time_lines)
        f.close()

    return bound_times_list
