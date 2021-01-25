# iterate through audio

from util import power, mix
from torch.utils.data import Dataset
import numpy as np
import pyroomacoustics as pra
import os

class OnlineSimulationDataset(Dataset):
    def __init__(self, voice_collection, noise_collection, length, simulation_config, truncator, cache_folder):
        self.voices=voice_collection
        self.noises=noise_collection
        self.length=length
        self.seed=simulation_config['seed']
        self.additive_noise_min_snr=simulation_config['min_snr']
        self.additive_noise_max_snr=simulation_config['max_snr']
        self.special_noise_ratio=simulation_config['special_noise_ratio']
        self.max_source=simulation_config['max_source']
        self.min_angle_diff=simulation_config['min_angle_diff']
        self.max_rt60=simulation_config['max_rt60'] # 0.3s
        self.min_rt60=0.15 # minimum to satisfy room odometry
        self.max_room_dim=simulation_config['max_room_dim'] # [10,10,4]
        self.min_room_dim=simulation_config['min_room_dim'] #[4,4,2]
        self.min_dist=simulation_config['min_dist'] # 0.8, dist between mic and person
        self.min_gap=simulation_config['min_gap'] # 1.2, gap between mic and walls
        self.max_order=simulation_config['max_order']
        self.randomize_material_ratio=simulation_config['randomize_material_ratio']
        self.max_latency=simulation_config['max_latency']
        self.random_volume_range=simulation_config['random_volume_range'] # max and min volume ratio for sources
        
        self.truncator=truncator
        self.cache_folder=cache_folder
        
    def __seed_for_idx(self,idx):
        return self.seed+idx
    
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
        
        if idx>=self.length:
            return None
        
        if self.cache_folder is not None:
            cache_path=self.cache_folder+'/'+str(idx)+'-'+str(self.seed)+'.npz'
            if os.path.exists(cache_path):
                cache_result=np.load(cache_path, allow_pickle=True)['data']
                return cache_result[0], cache_result[1], cache_result[2], cache_result[3], cache_result[4]
        else:
            cache_path=None
            
        np.random.seed(self.__seed_for_idx(idx))
        n_source=np.random.randint(self.max_source)+1
        
        room_result=simulateRoom(n_source, self.min_room_dim, self.max_room_dim, self.min_gap, self.min_dist, self.min_angle_diff)
        if room_result is None:
            return self.__getitem(idx+1) # backoff
            
        room_dim, R_loc, source_loc, source_angles=room_result
        
        voices=[self.truncator.process(self.voices[vi]) for vi in np.random.choice(len(self.voices), n_source)]
        voices=[v*np.random.uniform(self.random_volume_range[0], self.random_volume_range[1]) for v in voices]
        
        
        if self.special_noise_ratio<np.random.rand():
            noise=self.truncator.process(self.noises[np.random.choice(len(self.noises))])
        else:
            noise=np.random.randn(self.truncator.get_length())

        if self.randomize_material_ratio<np.random.rand():
            ceiling, east, west, north, south = tuple(np.random.choice(wall_materials, 5))  # sample material
            floor = np.random.choice(floor_materials)  # sample material
            mixed, premix_w_reverb, premix=simulateSound(room_dim, R_loc, source_loc, voices, 0, (ceiling, east, west, north, south, floor), self.max_order)
        else:
            rt60=np.random.uniform(self.min_rt60, self.max_rt60)
            mixed, premix_w_reverb, premix=simulateSound(room_dim, R_loc, source_loc, voices, rt60)
        
        
        background=simulateBackground(noise)
        snr=np.random.uniform(self.additive_noise_min_snr, self.additive_noise_max_snr)
        
        # trucate to the same length
        mixed=mixed[:, :truncator.get_length()]
        background=background[:, :truncator.get_length()]
        
        total, background=mix(mixed, background, snr)
        
        # save cache
        np.savez_compressed(cache_path, data=[total, premix_w_reverb, source_angles, premix, background])
        
        return total, premix_w_reverb, source_angles, premix, background
    