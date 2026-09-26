import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import os
import zarr
import time
from multiprocessing import Process, Queue
from communicate.envonline3cleanHSKE_XYXbot import USForceOnlineRead


class USSimpleImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            **kwargs # 新加的
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1) # fps = 10
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        for i in range(n_train): # n_train = 6
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn)) 

        for i in range(n_test): # n_test = 50
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        env = None
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        if 'classifier' in kwargs:
            self.classifier = kwargs['classifier']
        else:
            self.classifier = None


    def run(self, policy: BaseImagePolicy,fpcheckpoint=None, policy2 = None, policy3 = None):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        load_previous = False
        print('********attention********')
        print('loadprevious:',load_previous)
        self.UsePolicy2 = r'C:\Users\xuyixiao\Desktop\Seafile\私人资料库\和子涵的paper\CODE\Policy\Marksave\UsePolicy2.txt'
        self.UsePolicy3 = r'C:\Users\xuyixiao\Desktop\Seafile\私人资料库\和子涵的paper\CODE\Policy\Marksave\UsePolicy3.txt'
        if os.path.exists(self.UsePolicy2):
            os.remove(self.UsePolicy2)
        if os.path.exists(self.UsePolicy3):
            os.remove(self.UsePolicy3)


        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs
            input("▶  Ready when you are—press <Enter> to start the policies…")
            env = USForceOnlineRead(dpbuffer='./tmp',fpcheckpoint=fpcheckpoint,load_previous=load_previous)
            env.setup(self.n_obs_steps)
            if self.classifier is not None:
                env.setup_classifier(self.classifier)
            obs, reward, done, info = env.step(np.zeros([1,8,12]),init=True,load_previous=load_previous) # very important
            past_action = None
            policy.reset()  
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            totalstep = 0
            while(1):
                if os.path.exists(self.UsePolicy2):
                    policy = policy2
                    print('using policy2')
                if os.path.exists(self.UsePolicy3):
                    policy = policy3
                    print('using policy3')
                totalstep += 1
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[:,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                obs_dict['action'] = obs_dict['action'].float()
                # run policy
                infstart = time.time()
                with torch.no_grad():
                    action_dict = policy.predict_action_conduct(obs_dict)
                print('第{totalstep}步推理完成，推理时间inference time:',time.time()-infstart,' ',time.time())
                # device_transfer
                np_action_dict = dict_apply(action_dict, 
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                six_forcebase = np.zeros([1,6]);rob_forcebase = np.zeros([1,6])
                # step env
                print('enter env step')
                obs, reward, done, info = env.step(action,robforcebase=rob_forcebase,sixforcebase=six_forcebase,obs_dict=obs_dict,load_previous=load_previous,topindices = np_action_dict['topindices'],lastobs = obs) # action [56,8,2] 现在问题是为什么会自己创建
                # classifier done or not
                done_start_time = time.time()
                with torch.no_grad():
                    done = self.classifier.predict_action({'obs':{'image':obs['obs']['image'][0]}})['pred']>0.9
                if done:
                    if not os.path.exists(self.UsePolicy2):
                        with open(self.UsePolicy2,'w') as f:
                            f.write('1')
                    else:
                        with open(self.UsePolicy3,'w') as f:
                            f.write('1')

                print('done time:',time.time()-done_start_time,' ',time.time())
                done = np.all(done.detach().cpu().numpy())
                past_action = action
                # update pbar
                print(f'第{totalstep}步完全执行完成')
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
