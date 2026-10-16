from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST, iTransformer, TimeVLM
from modules import get_wrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler # [추가] 스케줄러 사용을 위한 임포트
import os
import time
import warnings
import numpy as np
# import wandb
import yaml
import matplotlib.pyplot as plt
from contextlib import nullcontext

warnings.filterwarnings('ignore')

MODEL_DICT = {
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'TimeVLM': TimeVLM, # TimeVLM 모델은 pretraining에 구현되어 있지 않으므로 None으로 설정
}

class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)
        
    def _build_model(self):
        if self.args.model not in MODEL_DICT:
            raise NotImplementedError(f"Model {self.args.model} is not implemented for pretraining.")

        backbone = MODEL_DICT[self.args.model].Model(self.args).float()
        wrapper = get_wrapper(self.args, backbone).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            wrapper = nn.DataParallel(wrapper, device_ids=self.args.device_ids)

        model = wrapper # wrapper 대신 바로 model이라는 이름을 사용할 수 있지만, 나중에 backbone과 wrapper를 헷갈릴 것 같아 명시적으로 wrapper라는 단어 사용 후 model이라는 변수명 사용(exp_main과 동일한 양식 사용).
        if self.args.torch_compile:
            try:
                print("Attempting to compile the model with torch.compile...")
                model = torch.compile(model)
            except Exception as e:
                print(f"torch.compile failed: {e}. Continuing without compilation.")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, device=self.device)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def cal_contrastive_loss(self, pred, target, temperature=0.5):
        B = pred.shape[0]
        
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)
        
        logits = torch.matmul(pred, target.T) / temperature
        labels = torch.arange(B).to(pred.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def train(self, setting):
        # Initialize wandb
        # try:
        #     wandb.init(
        #         project="OilO-Pretrain",
        #         name=setting,
        #         tags=[self.args.des] if self.args.des else [],
        #         config=vars(self.args)
        #     )
        # except Exception as e:
        #     print(f"wandb init failed: {e}. Continuing without wandb logging.")
        
        train_data, train_loader = self._get_data(flag='pretrain')
        
        path = os.path.join(self.args.pretrain_results_path, self.args.model, self.args.data, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # [수정] history 딕셔너리를 통째로 저장
        # 로드 예시: data = np.load('history.npz'); lr = data['learning_rates']
        with open(os.path.join(path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, indent=4)
            
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()

        # [추가] Learning Rate Scheduler 설정
        # Pre-training에서는 보통 Cosine Annealing을 많이 사용하여 학습 후반부에 LR을 부드럽게 줄입니다.
        scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs, eta_min=1e-6)
        
        # [수정] 변수명을 loss_history -> history로 변경 (확장성 고려)
        # 'learning_rates' 키를 추가하여 LR 변화 기록
        history = {'learning_rates': []}
        
        min_loss = float('inf')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_loss_accum = {} 
            
            self.model.train()
            epoch_time = time.time()
            
            # [기록] 현재 Epoch 시작 시점의 Learning Rate 저장
            current_lr = model_optim.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
                
            if self.args.use_amp:
                context = torch.autocast(device_type='cuda', dtype=torch.float16) # 혹은 bfloat16
            else:
                context = nullcontext()

            for i, (batch_x_long, batch_x_mark_long) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                
                # Random Tau Sampling
                tau = np.random.randint(1, self.args.max_tau + 1)
                
                batch_x_long = batch_x_long.float()
                batch_x_mark_long = batch_x_mark_long.float()
                if not self.args.gpu_preload:
                    batch_x_long = batch_x_long.to(self.device, non_blocking=True)
                    batch_x_mark_long = batch_x_mark_long.to(self.device, non_blocking=True)
                    
                x_t = batch_x_long[:, :self.args.seq_len, :]
                x_tau = batch_x_long[:, tau: self.args.seq_len + tau, :]
                x_t_mark = batch_x_mark_long[:, :self.args.seq_len, :]
                x_tau_mark = batch_x_mark_long[:, tau: self.args.seq_len + tau, :]
                

                with context:
                    if self.args.module_version == 'timesiam':
                        outputs = self.model(x_t, x_tau, tau, x_t_mark=x_t_mark, x_tau_mark=x_tau_mark, pretrain=True)
                    else:
                        outputs = self.model(x_t, x_tau, tau, x_t_mark=x_t_mark, x_tau_mark=x_tau_mark)
                        
                    
                    loss_dict = self.model.loss_fn(outputs)
                    loss = loss_dict['loss']
                    
                    for key, val in loss_dict.items():
                        epoch_loss_accum.setdefault(key, []).append(val.detach())
                
                self.scaler.scale(loss).backward()
                self.scaler.step(model_optim)
                self.scaler.update()

                
                if (i + 1) % 100 == 0:
                    current_loss = loss.item()
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | tau: {3} | lr: {4:.7f}".format(
                        i + 1, epoch + 1, current_loss, tau, current_lr))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    iter_count = 0
                    time_now = time.time()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # Epoch 종료 후 Scheduler Step (다음 Epoch를 위한 LR 조정)
            scheduler.step()
            
            # 에포크 평균 계산 및 History 저장
            # epoch_avg_losses = {key: np.average(val) for key, val in epoch_loss_accum.items()}
            epoch_avg_losses = {key: torch.stack(val).mean().item() for key, val in epoch_loss_accum.items()}
            
            for key, val in epoch_avg_losses.items():
                if key not in history: # loss 키가 없으면 생성
                    history[key] = []
                history[key].append(val)
                
            current_total_loss = epoch_avg_losses['loss']
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Next LR: {3:.7f}".format(
                epoch + 1, train_steps, current_total_loss, model_optim.param_groups[0]['lr']))
            
            # Best Model 저장 (Total Loss 기준)
            if current_total_loss < min_loss:
                print(f"Loss decreased ({min_loss:.6f} --> {current_total_loss:.6f}). Saving model ...")
                
                # VLM 제외 여부 결정: finetune_vlm=False일 때만 제외
                exclude_vlm = not getattr(self.args, 'finetune_vlm', False)
                state_dict = self.model.state_dict()
                
                if exclude_vlm:
                    from utils.tools import filter_vlm_from_state_dict
                    filtered_state_dict, excluded_keys = filter_vlm_from_state_dict(state_dict, exclude_vlm=True)
                    print(f"Excluding {len(excluded_keys)} VLM parameters from checkpoint (saving {len(filtered_state_dict)} parameters)")
                    torch.save(filtered_state_dict, path + '/' + 'checkpoint.pth')
                else:
                    print(f"Including all {len(state_dict)} parameters (VLM will be saved)")
                    torch.save(state_dict, path + '/' + 'checkpoint.pth')
                
                min_loss = current_total_loss
        
        # 빈 .finish 파일 생성 (실험 완료 표시용)
        with open(os.path.join(path, '.finish'), 'w') as f:
            f.write('')
        np.savez(os.path.join(path, 'history.npz'), **history)

        # try:
        #     wandb.save(os.path.join(path, 'history.npz'))
        #     # Finish wandb run
        #     wandb.finish()
        # except Exception as e:
        #     pass

        return self.model