from __future__ import division
import os


print(os.getcwd())
import ssl


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


vctk_path = '/data1/VCTK-Corpus'
locata_path = '../remote'
cache_path = '../cache'
fs = 48000
V = 343



# locata=LocataDataset('../remote/eval/', 'benchmark2', 48000, dev=True)

vctk = torchaudio.datasets.VCTK(root="/data1/", folder_in_archive="VCTK-Corpus")

print(len(vctk))
print(vctk[0][0].numpy()[0])


class VCTKAudio:
    def __init__(self, vctk, numpy=True):
        self.vctk = vctk
        self.numpy = numpy

    def __getitem__(self, idx):
        return self.vctk[idx][0][0].numpy() if self.numpy else self.vctk[idx][0][0]

    def __len__(self):
        return len(self.vctk)


import librosa


class MS_SNSD:
    def __init__(self, path, shuffle=True):
        noisy_wav_dir_path = [path + '/noise_test', path + '/noise_train']
        self.noisy_files = []
        for noisy_wav_dir in noisy_wav_dir_path:
            for file in os.listdir(noisy_wav_dir):
                if file.endswith('.wav'):
                    self.noisy_files.append(os.path.join(noisy_wav_dir, file))
        if shuffle:
            np.random.shuffle(self.noisy_files)

    def __len__(self):
        return len(self.noisy_files)

    def __read_audio(self, file):
        noise_audio, _ = librosa.load(file, sr=fs, mono=True)
        return noise_audio

    def __getitem__(self, idx):
        return self.__read_audio(self.noisy_files[idx])

    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        for idx in range(idx1, idx2):
            mic_sig_batch.append(self.__read_audio(self.noisy_files[idx]))

        return np.stack(mic_sig_batch)


ms_snsd = MS_SNSD('/data1/MS-SNSD')

# acoustic property
N_MIC = 6
R_MIC = 0.06

max_rt60 = 0.3
max_room_dim = [10, 10, 4]
min_room_dim = [4, 4, 2]
min_dist = 0.8  # dist between mic and person
min_gap = 1.2  # gap between mic and walls

import pyroomacoustics as pra


def generate_mic_array(mic_radius: float, n_mics: int, pos):
    """
    Generate a list of Microphone objects
    Radius = 50th percentile of men Bitragion breadth
    (https://en.wikipedia.org/wiki/Human_head)
    """
    R = pra.circular_2D_array(center=[pos[0], pos[1]], M=n_mics, phi0=0, radius=mic_radius)
    R = np.concatenate((R, np.ones((1, n_mics)) * pos[2]), axis=0)
    return R


# global mic array:
R_global = generate_mic_array(R_MIC, N_MIC, (0, 0, 0))


# simulate the room

def get_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_angle(px, py):
    return np.arctan2(py, px)


def random2D(range_x, range_y, except_x, except_y, except_r, retry=100):
    if retry == 0: return None
    if except_r < except_x < range_x - except_r and except_r < except_y < range_y - except_r:
        loc = (np.random.uniform(0, range_x), np.random.uniform(0, range_y))
        if get_dist(loc, (except_x, except_y)) < except_r:
            return random2D(range_x, range_y, except_x, except_y, except_r, retry - 1)
        else:
            return loc
    else:
        return None


def simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff=0, retry=15):
    if retry == 0: return None
    # return simulated room
    room_dim = [np.random.uniform(low, high) for low, high in zip(min_room_dim, max_room_dim)]
    R_loc = [np.random.uniform(min_gap, x - min_gap) for x in room_dim]
    source_locations = [random2D(room_dim[0], room_dim[1], R_loc[0], R_loc[1], min_dist) for i in range(N_source)]
    if None in source_locations: return None

    angles = [get_angle(p[0] - R_loc[0], p[1] - R_loc[1]) for p in source_locations]
    if N_source > 1:
        min_angle_diff *= np.pi / 180
        angles_sorted = np.sort(angles)
        if np.min(angles_sorted[1:] - angles_sorted[:-1]) < min_angle_diff or angles_sorted[0] - angles_sorted[
            -1] + 2 * np.pi < min_angle_diff:
            return simulateRoom(N_source, min_room_dim, max_room_dim, min_gap, min_dist, min_angle_diff, retry - 1)

    source_locations = [(x, y, R_loc[2]) for x, y in source_locations]

    return (room_dim, R_loc, source_locations, angles)


# Define materials
wall_materials = [
    'hard_surface',
    'brickwork',
    'rough_concrete',
    'unpainted_concrete',
    'rough_lime_wash',
    'smooth_brickwork_flush_pointing',
    'smooth_brickwork_10mm_pointing',
    'brick_wall_rough',
    'ceramic_tiles',
    'limestone_wall',
    'reverb_chamber'
]


floor_materials = [
    'linoleum_on_concrete',
    'carpet_cotton',
    'carpet_tufted_9.5mm',
    'carpet_thin',
    'carpet_6mm_closed_cell_foam',
    'carpet_6mm_open_cell_foam',
    'carpet_tufted_9m',
    'felt_5mm',
    'carpet_soft_10mm',
    'carpet_hairy',
]


def simulateSound(room_dim, R_loc, source_locations, source_audios, rt60, materials=None, max_order=None):
    # source_audios: array of numpy array
    # L: max of all audios. Zero padding at the end
    # return (all_channel_data (C, L), groundtruth_with_reverb (N, C, L), groundtruth_data (N, C, L), angles (N)

    if materials is not None:
        (ceiling, east, west, north, south, floor) = materials
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.make_materials(
                ceiling=ceiling,
                floor=floor,
                east=east,
                west=west,
                north=north,
                south=south,
            ), max_order=max_order
        )
    else:
        try:
            e_absorption, max_order_rt60 = pra.inverse_sabine(rt60, room_dim)
        except ValueError:
            e_absorption, max_order_rt60 = pra.inverse_sabine(1, room_dim)
        room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order_rt60)

    R = generate_mic_array(R_MIC, N_MIC, R_loc)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    length = max([len(source_audios[i]) for i in range(len(source_audios))])
    for i in range(len(source_audios)):
        source_audios[i] = np.pad(source_audios[i], (0, length - len(source_audios[i])), 'constant')

    for i in range(len(source_locations)):
        room.add_source(source_locations[i], signal=source_audios[i], delay=0)

    room.image_source_model()
    premix_w_reverb = room.simulate(return_premix=True)
    mixed = room.mic_array.signals

    # groundtruth
    room_gt = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(1.0), max_order=0)
    # R_gt=generate_mic_array(R_MIC, N_MIC, R_loc)
    R_gt = generate_mic_array(0, 1, R_loc)
    room_gt.add_microphone_array(pra.MicrophoneArray(R_gt, room.fs))

    for i in range(len(source_locations)):
        room_gt.add_source(source_locations[i], signal=source_audios[i], delay=0)
    room_gt.compute_rir()

    room_gt.image_source_model()
    premix = room_gt.simulate(return_premix=True)

    return (mixed, premix_w_reverb, premix, R)


def simulateBackground(background_audio):
    # diffused noise. simulate in a large room
    bg_radius = np.random.uniform(low=10.0, high=20.0)
    bg_theta = np.random.uniform(low=0, high=2 * np.pi)
    H = 10
    bg_loc = [bg_radius * np.cos(bg_theta), bg_radius * np.sin(bg_theta), H]

    # Bg should be further away to be diffuse
    left_wall = np.random.uniform(low=-40, high=-20)
    right_wall = np.random.uniform(low=20, high=40)
    top_wall = np.random.uniform(low=20, high=40)
    bottom_wall = np.random.uniform(low=-40, high=-20)
    height = np.random.uniform(low=20, high=40)
    corners = np.array([[left_wall, bottom_wall], [left_wall, top_wall],
                        [right_wall, top_wall], [right_wall, bottom_wall]]).T
    absorption = np.random.uniform(low=0.5, high=0.99)
    room = pra.Room.from_corners(corners,
                                 fs=fs,
                                 max_order=10,
                                 materials=pra.Material(absorption))
    room.extrude(height)
    mic_array = generate_mic_array(R_MIC, N_MIC, (0, 0, H))
    room.add_microphone_array(pra.MicrophoneArray(mic_array, fs))
    room.add_source(bg_loc, signal=background_audio)

    room.image_source_model()
    room.simulate()
    return room.mic_array.signals


room_dim, R_loc, source_loc, angles = simulateRoom(3, (3, 3, 3), (10, 10, 10), 1, 1, 10)

mixed, premix_w_reverb, premix, R = simulateSound(room_dim, R_loc, source_loc,
                                                  [np.random.randn(10000) for i in range(len(source_loc))], 0.3)

print(mixed.shape, premix.shape, premix_w_reverb.shape)

ceiling, east, west, north, south = tuple(np.random.choice(wall_materials, 5))  # sample material
floor = np.random.choice(floor_materials)  # sample material
mixed, premix_w_reverb, premix, R = simulateSound(room_dim, R_loc, source_loc,
                                                  [np.random.randn(10000) for i in range(len(source_loc))], 0,
                                                  (ceiling, east, west, north, south, floor), 5)

print(mixed.shape, premix.shape, premix_w_reverb.shape)

# iterate through audio

from util import power, mix
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import os


class OnlineSimulationDataset(Dataset):
    def __init__(self, voice_collection, noise_collection, length, simulation_config, truncator, cache_folder,
                 cache_max=None):
        self.voices = voice_collection
        self.noises = noise_collection
        self.length = length
        self.seed = simulation_config['seed']
        self.additive_noise_min_snr = simulation_config['min_snr']
        self.additive_noise_max_snr = simulation_config['max_snr']
        self.special_noise_ratio = simulation_config['special_noise_ratio']
        self.max_source = simulation_config['max_source']
        self.min_angle_diff = simulation_config['min_angle_diff']
        self.max_rt60 = simulation_config['max_rt60']  # 0.3s
        self.min_rt60 = 0.15  # minimum to satisfy room odometry
        self.max_room_dim = simulation_config['max_room_dim']  # [10,10,4]
        self.min_room_dim = simulation_config['min_room_dim']  # [4,4,2]
        self.min_dist = simulation_config['min_dist']  # 0.8, dist between mic and person
        self.min_gap = simulation_config['min_gap']  # 1.2, gap between mic and walls
        self.max_order = simulation_config['max_order']
        self.randomize_material_ratio = simulation_config['randomize_material_ratio']
        self.max_latency = simulation_config['max_latency']
        self.random_volume_range = simulation_config['random_volume_range']  # max and min volume ratio for sources

        self.truncator = truncator
        self.cache_folder = cache_folder
        self.cache_history = []
        self.cache_max = cache_max

    def __seed_for_idx(self, idx):
        return self.seed + idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return format:
        # (
        # mixed multichannel audio, (C,L)
        # array of groundtruth with reverb for each target, (N, C, L)
        # array of direction of targets, (N,)
        # array of multichannel ideal groundtruths for each target, (N, C, L)
        # noise (C, L)
        # )
        # check cache first

        if idx >= self.length:
            return None

        if self.cache_folder is not None:
            cache_path = self.cache_folder + '/' + str(idx) + '-' + str(self.seed) + '.npz'

            if cache_path not in self.cache_history:
                self.cache_history.append(cache_path)

                if self.cache_max is not None and self.cache_max == len(self.cache_history):
                    # delete first one
                    first = self.cache_history[0]
                    os.remove(first)
                    self.cache_history = self.cache_history[1:]

            if os.path.exists(cache_path):
                cache_result = np.load(cache_path, allow_pickle=True)['data']
                return cache_result[0], cache_result[1], cache_result[2], cache_result[3], cache_result[4]
        else:
            cache_path = None

        np.random.seed(self.__seed_for_idx(idx))
        # n_source=np.random.randint(self.max_source)+1
        n_source = 3

        room_result = simulateRoom(n_source, self.min_room_dim, self.max_room_dim, self.min_gap, self.min_dist,
                                   self.min_angle_diff)
        if room_result is None:
            return self.__getitem__(idx + 1)  # backoff

        room_dim, R_loc, source_loc, source_angles = room_result

        voices = [self.truncator.process(self.voices[vi]) for vi in np.random.choice(len(self.voices), n_source)]
        voices = [v * np.random.uniform(self.random_volume_range[0], self.random_volume_range[1]) for v in voices]

        if self.special_noise_ratio < np.random.rand():
            noise = self.truncator.process(self.noises[np.random.choice(len(self.noises))])
        else:
            noise = np.random.randn(self.truncator.get_length())

        if self.randomize_material_ratio < np.random.rand():
            ceiling, east, west, north, south = tuple(np.random.choice(wall_materials, 5))  # sample material
            floor = np.random.choice(floor_materials)  # sample material
            mixed, premix_w_reverb, premix, R = simulateSound(room_dim, R_loc, source_loc, voices, 0,
                                                              (ceiling, east, west, north, south, floor),
                                                              self.max_order)
        else:
            rt60 = np.random.uniform(self.min_rt60, self.max_rt60)
            mixed, premix_w_reverb, premix, R = simulateSound(room_dim, R_loc, source_loc, voices, rt60)

        background = simulateBackground(noise)
        snr = np.random.uniform(self.additive_noise_min_snr, self.additive_noise_max_snr)

        # trucate to the same length
        mixed = mixed[:, :truncator.get_length()]
        background = background[:, :truncator.get_length()]

        total, background = mix(mixed, background, snr)

        # save cache
        if cache_path is not None:
            np.savez_compressed(cache_path, data=[total, premix_w_reverb, source_angles, premix, background])
        if premix_w_reverb.shape[-1] <= 18000:
            offset = 18000 - premix_w_reverb.shape[-1]
            premix_w_reverb = np.pad(premix_w_reverb, ((0, 0), (0, 0), (0, offset)), "constant")
        else:
            premix_w_reverb = premix_w_reverb[..., :18000]

        return total, premix_w_reverb, source_angles, premix, background, R, R_loc


# randomly truncate audio to fixed length

class RandomTruncate:
    def __init__(self, target_length, seed, power_threshold=None):
        self.length = target_length
        self.seed = seed
        self.power_threshold = power_threshold
        np.random.seed(seed)

    def process(self, audio):
        # if there is a threshold
        if self.power_threshold is not None:
            # smooth
            power = np.convolve(audio ** 2, np.ones((32,)), 'same')
            avgpower = np.mean(power)
            for i in range(len(power)):
                # threshold*mean_power
                if power[i] > avgpower * self.power_threshold:
                    # print(i, power[i], avgpower)
                    # leave ~=0.3s of start
                    fs = 48000
                    audio = audio[max(0, i - int(0.3 * fs)):]
                    break

        if len(audio) < self.length:
            nfront = np.random.randint(self.length - len(audio))
            return np.pad(audio, (nfront, self.length - len(audio) - nfront), 'constant')
        elif len(audio) == self.length:
            return audio
        else:
            start = np.random.randint(len(audio) - self.length)
            return audio[start:start + self.length]

    def get_length(self):
        return self.length


simulation_config = {
    'seed': 3,
    'min_snr': 25,
    'max_snr': 40,
    'special_noise_ratio': 0.5,
    'max_source': 3,
    'min_angle_diff': 15,
    'max_rt60': 0.3,
    'max_room_dim': [10, 10, 4],
    'min_room_dim': [4, 4, 2],
    'min_dist': 0.8,
    'min_gap': 1.2,
    'max_order': 5,
    'randomize_material_ratio': 0.5,
    'max_latency': 0.5,
    'random_volume_range': [0.7, 1]
}

simulation_config_test={
    'seed':5,
    'min_snr':25,
    'max_snr':40,
    'special_noise_ratio':0.5,
    'max_source':3,
    'min_angle_diff':15,
    'max_rt60': 0.3,
    'max_room_dim':[10,10,4],
    'min_room_dim':[4,4,2],
    'min_dist': 0.8,
    'min_gap': 1.2,
    'max_order':5,
    'randomize_material_ratio':0.5,
    'max_latency':0.5,
    'random_volume_range': [0.7, 1]
}
vctk_audio = VCTKAudio(vctk)
truncator = RandomTruncate(3 * fs, 5, 0.4)
dataset = OnlineSimulationDataset(vctk_audio, ms_snsd, 150, simulation_config, truncator, None, 100)

R_global=generate_mic_array(R_MIC, N_MIC, (0,0,0))

if __name__ == "__main__":
    for i in range(10):
        data = dataset.__getitem__(i)
        print(data[0].shape)


