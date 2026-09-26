import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-l', '--classification_checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint,classification_checkpoint, output_dir, device):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Offline: 
    # load the checkpoint of classfifier
    # payload = torch.load(open(classification_checkpoint, 'rb'), pickle_module=dill)
    # cfg = payload['cfg']
    # cls = hydra.utils.get_class(cfg._target_)
    # workspace = cls(cfg, output_dir=output_dir)
    # workspace: BaseWorkspace
    # workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # # get classifier from workspace
    # classifier = workspace.model
    # if cfg.training.use_ema:
    #     classifier = workspace.ema_model
    
    # device = torch.device(device)
    # classifier.to(device)
    # classifier.eval()
    classifier = None

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace 
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    

    # run eval
    print(cfg.task.env_runner)
    cfg.task.env_runner['_target_'] = 'diffusion_policy.env_runner.USSimple_image_runner_offline.USSimpleImageRunner'
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        classifier=classifier,)
    runner_log = env_runner.run(policy,fpcheckpoint=checkpoint)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
