from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer
from concurrent.futures import ThreadPoolExecutor  


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, # sequence_length: e.g., 16
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]: # test就跳过了
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1] # 设置上一帧结束为轨迹的起点
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

# def get_test_mask(n_episodes, test_ratio, seed=0):
#     test_mask = np.zeros(n_episodes, dtype=bool)
#     if test_ratio <= 0:
#         return test_mask
    
#     # have at least 1 episode for validation, and at least 2 episode for train
#     n_test = min(max(1, round(n_episodes * test_ratio)), n_episodes-1)
#     rng = np.random.default_rng(seed=seed)
#     test_idxs = rng.choice(n_episodes, size=n_test, replace=False)
#     test_mask[test_idxs] = True
#     return test_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data 
    # zhthink: 比如原来轨迹mask有205个，现在只用其中90个来练
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends,  # 看上去是获得每个episode的开始和结束的index
                sequence_length=sequence_length, # e.g., 16
                pad_before=pad_before,  # 1
                pad_after=pad_after,    # 7
                episode_mask=episode_mask # indices.shape = [10726,4]
                )
            '''
            indices.append([
            buffer_start_idx, buffer_end_idx, 
            sample_start_idx, sample_end_idx])
            看上去是对于一些轨迹的起点要pad一格
            Example of the indices:
            array([[964, 979,   1,  16],
                    [964, 980,   0,  16],
                    [965, 981,   0,  16],
                    [966, 982,   0,  16],
                    [967, 983,   0,  16],
                    [968, 984,   0,  16],
                    [969, 985,   0,  16],
                    [970, 986,   0,  16],
                    [971, 987,   0,  16],
                    [972, 988,   0,  16],
                    [973, 989,   0,  16],
                    [974, 990,   0,  16],
                    [975, 991,   0,  16],
                    [976, 992,   0,  16],
                    [977, 993,   0,  16],
                    [978, 994,   0,  16],
                    [979, 995,   0,  16],
                    [980, 996,   0,  16],
                    [981, 997,   0,  16],
                    [982, 998,   0,  16]])
        ([    0,   162,   289,   470,   612,   730,   849,   984,  1110,
            '''
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                if key == 'img' and type(input_arr[buffer_start_idx]) == np.str_:      
                    sample = load_data_in_parallel(input_arr[buffer_start_idx:buffer_end_idx])
                else:
                    sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                if key == 'img' and type(input_arr[buffer_start_idx]) == np.str_:
                    data = np.zeros(
                        shape=(self.sequence_length,) + sample.shape[1:],
                        dtype=input_arr.dtype)   
                else:
                    data = np.zeros(
                        shape=(self.sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype)                    
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
    
def load_file(filename):  
    return np.load(filename)  

def load_data_in_parallel(filenames):  
    with ThreadPoolExecutor() as executor:  
        samples = list(executor.map(load_file, filenames))  
    return np.stack(samples)  