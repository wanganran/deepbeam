beamformer_online_conf = {'sr': 24000,   # sample rate
                   'ban': False,
                   'vad_proportion': 1,  # Energy proportion to filter silence masks [0.5, 1]
                   'alpha': 0.8,  # Online Remember coefficient when updating covariance matrix
                   'chunk_size': 50,
                   'channels': 6,  # Number of channels available
                   'mask_alpha': False,
                   'num_iters': 10,
                   'solve_permu': False  # If true, solving permutation problems
                   }
stft_conf = {'frame_len': 1024,
             'frame_hop':256,
             "round_power_of_two": True,
             "window": 'hann',
             "center":False,
             "transpose": False
             }
import numpy as np
import importlib
import apply_online_beamformer
importlib.reload(apply_online_beamformer)
from apply_online_beamformer import apply_online_beamformer


cache_folder='../../bf_cache'
def getData(idx):
    cache_path=cache_folder+'/sbd-'+str(idx)+'.npz'
    cache_result=np.load(cache_path, allow_pickle=True)['data']
    total, bfdata, gt, source_angles = cache_result[0], cache_result[1], cache_result[2], cache_result[3]
    return total, gt

signal, gt=getData(1)
sigresult=apply_online_beamformer(signal, stft_conf, beamformer_online_conf)
print(sigresult.shape)