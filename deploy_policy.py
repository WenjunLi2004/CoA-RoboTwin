import numpy as np
import torch
import os
import sys
import cv2
import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as transforms
from argparse import Namespace
import torchvision.transforms.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # 添加当前目录
sys.path.insert(0, os.path.join(current_dir, "src"))  # 添加src目录

from src.methods.coa import CoA
from src.methods.utils import *
from src.workspace import Workspace
from src.methods.base import BaseMethod
from src.envs.rlbench.wrappers.rescale_from_tanh import MinMaxNorm
from src.envs.rlbench.wrappers.rescale_from_tanh import get_action_space_from_cfg
from accelerate import Accelerator, DistributedDataParallelKwargs

def encode_obs(observation):  # 后处理观测数据
    processed_images = {}
    obs = {}
    # 摄像头映射：环境键名 -> CoA期望的键名
    camera_mapping = {
        "head_camera": "overhead_rgb",
        "left_camera": "left_shoulder_rgb", 
        "right_camera": "right_shoulder_rgb",
        "front_camera": "front_rgb"
    }
        
    # 处理每个摄像头的图像
    for env_key, coa_key in camera_mapping.items():
        if env_key in observation["observation"]:
            img = observation["observation"][env_key]["rgb"]
            # 转换为 PyTorch 张量
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).float()
            else:
                img_tensor = img.float()
                
            # 确保通道在最后：(H, W, C)
            if img_tensor.shape[0] == 3:  # 如果是 (C, H, W)
                img_tensor = img_tensor.permute(1, 2, 0)  # -> (H, W, C)
                             
            # 调整到模型期望的分辨率 128×128
            # 先转换为 (C, H, W) 格式用于 resize
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
            # 使用双线性插值调整大小
            img_resized = F.resize(img_tensor, (128, 128), antialias=True)
                
            # 添加批处理维度: (C, H, W) -> (1, C, H, W)
            img_resized = img_resized.unsqueeze(0)
                
            obs[coa_key] = img_resized
     
    state_data = observation["endpose"]["left_endpose"] + [observation["endpose"]["left_gripper"]] 
    
    state_tensor = torch.tensor(state_data, dtype=torch.float32)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)  # 添加批处理维度
    if state_tensor.dim() == 2:
        state_tensor = state_tensor.unsqueeze(1) # (batch, state_dim) -> (batch, 1, state_dim)
    obs["low_dim_state"] = state_tensor
        
    return obs


def get_model(usr_args):  # 来自 deploy_policy.yml 和 eval.sh 的覆盖参数
    # 直接使用 OmegaConf 加载配置文件
    config_path = Path(current_dir) / "deploy_policy.yml"

    cfg = OmegaConf.load(config_path)

    work_dir = Path(cfg.work_dir)
    
    correct_ckpt_path = work_dir / "checkpoints" / "coa_20000.pt"
        
    checkpoint = torch.load(correct_ckpt_path, map_location='cpu')

    cfg_ckpt = checkpoint["config"]
    cfg.env.env_name = cfg_ckpt.env.env_name
    cfg.method = cfg_ckpt.method
    cfg.action_sequence = cfg_ckpt.action_sequence
    cfg.method_name = cfg_ckpt.method_name

    state_dict = checkpoint['agent_state_dict']

    # 处理用户参数覆盖 - 支持多种格式
    if usr_args:
        if isinstance(usr_args, list):
            # 处理列表格式的参数
            for arg in usr_args:
                if isinstance(arg, str) and '=' in arg:
                    key, value = arg.split('=', 1)
                    # 尝试转换数值类型
                    if value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        value = float(value) if '.' in value else int(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    OmegaConf.set(cfg, key, value)
                    print(f"Set {key} = {value}")
        elif isinstance(usr_args, dict):
            # 处理字典格式
            cli_conf = OmegaConf.create(usr_args)
            cfg = OmegaConf.merge(cfg, cli_conf)
            print(f"Merged config with: {usr_args}")
   
    # 解析配置中的变量引用
    OmegaConf.resolve(cfg)
    
    # 修复OmegaConf将参数解析为String的问题
    def fix_numeric_types(config):
        if isinstance(config, DictConfig):
            for key, value in config.items():
                if isinstance(value, str):
                    if 'e-' in value or 'e+' in value or 'E-' in value or 'E+' in value:
                        try:
                            config[key] = float(value)
                        except ValueError:
                            pass
                    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                        config[key] = float(value) if '.' in value else int(value)
                elif isinstance(value, DictConfig):
                    fix_numeric_types(value)
    
    fix_numeric_types(cfg)
      
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    model = hydra.utils.instantiate(cfg.method,accelerator = accelerator)  
     
    model.load_state_dict(state_dict, strict=True)
    
    # 准备模型以进行评估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
        
    return model


def eval(TASK_ENV, model, observation):
    config_path = Path(os.path.dirname(os.path.abspath(__file__))) / "deploy_policy.yml"
    cfg = OmegaConf.load(config_path)

    obs = encode_obs(observation)  
    # instruction = TASK_ENV.get_instruction()
    
    # 将数据移动到模型的设备
    device = model.device
    batch_obs = obs
    batch_obs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v 
        for k, v in batch_obs.items()
    }     
    # 调用模型的 act 方法
    actions = model.act(batch_obs) 
    # 转换为 numpy 并移除批处理维度
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
            
    # 处理动作的维度
    if actions.ndim == 3:  # (batch, sequence, action_dim)
        actions = actions[0]  # 形状: (16,50)
    
    action_space = get_action_space_from_cfg(cfg)
    denormalized_action = MinMaxNorm.denormalize(actions, action_space)
    actions = denormalized_action
        
    for i,action in enumerate(actions):
        TASK_ENV.take_action(action, action_type='ee')  # 确保使用正确的action_type

def reset_model(model):
    pass