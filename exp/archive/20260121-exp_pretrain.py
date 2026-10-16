from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST
from modules import oilo as OilO 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler # [추가] 스케줄러 사용을 위한 임포트
import os
import time
import warnings
import numpy as np
import wandb
import yaml
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

MODEL_DICT = {
    'PatchTST': PatchTST
}

class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)
        
    def _build_model(self):
        if self.args.model not in MODEL_DICT:
            raise NotImplementedError(f"Model {self.args.model} is not implemented for pretraining.")

        backbone = MODEL_DICT[self.args.model].Model(self.args).float()
        wrapper = OilO.OilO_Wrapper(self.args, backbone).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            wrapper = nn.DataParallel(wrapper, device_ids=self.args.device_ids)

        model = wrapper # wrapper 대신 바로 model이라는 이름을 사용할 수 있지만, 나중에 backbone과 wrapper를 헷갈릴 것 같아 명시적으로 wrapper라는 단어 사용 후 model이라는 변수명 사용(exp_main과 동일한 양식 사용).
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
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
        wandb.init(
            project="OilO-Pretrain",
            name=setting,
            tags=[self.args.des] if self.args.des else [],
            config=vars(self.args)
        )
        
        train_data, train_loader = self._get_data(flag='pretrain')
        
        path = os.path.join(self.args.pretrain_results_path, self.args.model, self.args.data, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
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
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_loss_accum = {} 
            
            self.model.train()
            epoch_time = time.time()
            
            # [기록] 현재 Epoch 시작 시점의 Learning Rate 저장
            current_lr = model_optim.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            for i, (batch_long, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # Random Tau Sampling
                tau = np.random.randint(1, self.args.max_tau + 1)
                
                batch_long = batch_long.float().to(self.device)
                x_t = batch_long[:, :self.args.seq_len, :]
                x_tau = batch_long[:, tau: self.args.seq_len + tau, :]
                
                # Forward
                outputs = self.model(x_t, x_tau, tau)
                
                if self.args.contrastive_only:
                    z_tau, z_t = outputs['z_tau'], outputs['z_t']
                    loss = self.cal_contrastive_loss(z_t, z_tau, temperature=0.5)
                    epoch_loss_accum.setdefault('total_loss', []).append(loss.item())
                elif self.args.module_version == 'v1' or self.args.module_version == 'v1-recursive':
                    z_tau_pred = outputs['z_tau_pred']
                    z_tau = outputs['z_tau']
                    
                    z_t_pred = outputs['z_t_pred']
                    z_t = outputs['z_t']
                    
                    loss_contrastive_fwd = self.cal_contrastive_loss(z_tau_pred, z_tau, temperature=0.5)
                    loss_contrastive_bwd = self.cal_contrastive_loss(z_t_pred, z_t, temperature=0.5)
                    loss = (loss_contrastive_fwd + loss_contrastive_bwd) / 2
                    
                    for key in ['loss_fwd', 'loss_bwd', 'total_loss']:
                        if key not in epoch_loss_accum:
                            epoch_loss_accum[key] = []
                    epoch_loss_accum['loss_fwd'].append(loss_contrastive_fwd.item())
                    epoch_loss_accum['loss_bwd'].append(loss_contrastive_bwd.item())
                    epoch_loss_accum['total_loss'].append(loss.item())
                elif self.args.module_version == 'v1-stop-gradient':
                    z_tau_pred = outputs['z_tau_pred']
                    z_tau = outputs['z_tau']
                    
                    z_t_pred = outputs['z_t_pred']
                    z_t = outputs['z_t']
                    
                    loss_contrastive_fwd = (-F.cosine_similarity(z_tau_pred, z_tau.detach(), dim=-1).mean()) 
                    loss_contrastive_bwd = (-F.cosine_similarity(z_t_pred, z_t.detach(), dim=-1).mean()) 
                    loss = (loss_contrastive_fwd + loss_contrastive_bwd)
                    
                    for key in ['loss_fwd', 'loss_bwd', 'total_loss']:
                        if key not in epoch_loss_accum:
                            epoch_loss_accum[key] = []
                    epoch_loss_accum['loss_fwd'].append(loss_contrastive_fwd.item())
                    epoch_loss_accum['loss_bwd'].append(loss_contrastive_bwd.item())
                    epoch_loss_accum['total_loss'].append(loss.item())
                elif self.args.module_version == 'v1-triangular-causal':
                    loss_list = self.model.loss_fn(outputs)
                    loss = loss_list['loss']
                    loss_fwd = loss_list['loss_fwd']
                    loss_bwd = loss_list['loss_bwd']
                    
                    for key in ['loss_fwd', 'loss_bwd', 'total_loss']:
                        if key not in epoch_loss_accum:
                            epoch_loss_accum[key] = []
                    epoch_loss_accum['loss_fwd'].append(loss_fwd.item())
                    epoch_loss_accum['loss_bwd'].append(loss_bwd.item())
                    epoch_loss_accum['total_loss'].append(loss.item())
                else:
                    raise ValueError(f"Unknown module_version: {self.args.module_version}")
                    
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | tau: {3} | lr: {4:.7f}".format(
                        i + 1, epoch + 1, loss.item(), tau, current_lr))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    loss_log = {
                        key: np.average(val) for key, val in epoch_loss_accum.items()
                    }
                    
                    # Log to wandb
                    wandb.log({
                        'train/step_loss': loss.item(),
                        **{f'train/epoch_avg_{key}': val for key, val in loss_log.items()},
                        'train/tau': tau,
                        'train/learning_rate': current_lr
                    })
                    
                    iter_count = 0
                    time_now = time.time()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            # Epoch 종료 후 Scheduler Step (다음 Epoch를 위한 LR 조정)
            scheduler.step()
            
            # 에포크 평균 계산 및 History 저장
            epoch_avg_losses = {key: np.average(val) for key, val in epoch_loss_accum.items()}
            
            for key, val in epoch_avg_losses.items():
                if key not in history: # loss 키가 없으면 생성
                    history[key] = []
                history[key].append(val)
                
            current_total_loss = epoch_avg_losses['total_loss']
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Next LR: {3:.7f}".format(
                epoch + 1, train_steps, current_total_loss, model_optim.param_groups[0]['lr']))
            
            # Log epoch metrics to wandb
            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': current_total_loss,
                    'train/epoch_loss_fwd': epoch_avg_losses['loss_fwd'],
                    'train/epoch_loss_bwd': epoch_avg_losses['loss_bwd'],
                    'train/epoch_lr': model_optim.param_groups[0]['lr']
                })
            except:
                pass
            
            # Best Model 저장 (Total Loss 기준)
            if current_total_loss < min_loss:
                print(f"Loss decreased ({min_loss:.6f} --> {current_total_loss:.6f}). Saving model ...")
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                # wandb.save(path + '/' + 'checkpoint.pth')
                min_loss = current_total_loss

        # [수정] history 딕셔너리를 통째로 저장
        # 로드 예시: data = np.load('history.npz'); lr = data['learning_rates']
        with open(os.path.join(path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, indent=4)
        
        # 빈 .finish 파일 생성 (실험 완료 표시용)
        with open(os.path.join(path, '.finish'), 'w') as f:
            f.write('')
        np.savez(os.path.join(path, 'history.npz'), **history)
        
        # Load best model and visualize shift operators
        print("Loading best model for visualization...")
        self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))
        self.model.eval()
        
        # Extract shift operators from OilO wrapper
        # Handle DataParallel case
        if isinstance(self.model, nn.DataParallel):
            oilo_wrapper = self.model.module
        else:
            oilo_wrapper = self.model
        
        # Get shift parameters
        try:
            shift_fwd_vector = oilo_wrapper.S_fwd.shift_vector.detach().cpu().numpy()
            shift_fwd_matrix = oilo_wrapper.S_fwd.soft_matrix.detach().cpu().numpy()
            shift_bwd_vector = oilo_wrapper.S_bwd.shift_vector.detach().cpu().numpy()
            shift_bwd_matrix = oilo_wrapper.S_bwd.soft_matrix.detach().cpu().numpy()
            
            # Create heatmap visualizations
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Forward shift vector
            axes[0, 0].bar(range(len(shift_fwd_vector)), shift_fwd_vector)
            axes[0, 0].set_title('Forward Shift Vector', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Dimension')
            axes[0, 0].set_ylabel('Weight')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Forward soft matrix
            im1 = axes[0, 1].imshow(shift_fwd_matrix, cmap='RdBu_r', aspect='auto')
            axes[0, 1].set_title('Forward Soft Matrix', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Output Dimension')
            axes[0, 1].set_ylabel('Input Dimension')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Backward shift vector
            axes[1, 0].bar(range(len(shift_bwd_vector)), shift_bwd_vector)
            axes[1, 0].set_title('Backward Shift Vector', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Dimension')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Backward soft matrix
            im2 = axes[1, 1].imshow(shift_bwd_matrix, cmap='RdBu_r', aspect='auto')
            axes[1, 1].set_title('Backward Soft Matrix', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Output Dimension')
            axes[1, 1].set_ylabel('Input Dimension')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Save figure
            viz_dir = os.path.join(path, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            shift_viz_path = os.path.join(viz_dir, 'shift_operators.png')
            plt.savefig(shift_viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Shift operator visualization saved to {shift_viz_path}")
            
            # Save raw data as well
            np.savez(os.path.join(path, 'shift_operators.npz'),
                    shift_fwd_vector=shift_fwd_vector,
                    shift_fwd_matrix=shift_fwd_matrix,
                    shift_bwd_vector=shift_bwd_vector,
                    shift_bwd_matrix=shift_bwd_matrix)
            
            # Log to wandb
            wandb.log({
                "shift_operators": wandb.Image(shift_viz_path),
                "shift_fwd_vector_mean": shift_fwd_vector.mean(),
                "shift_fwd_vector_std": shift_fwd_vector.std(),
                "shift_bwd_vector_mean": shift_bwd_vector.mean(),
                "shift_bwd_vector_std": shift_bwd_vector.std(),
            })
            
            wandb.save(os.path.join(path, 'shift_operators.npz'))
            wandb.save(shift_viz_path)
        except:
            pass
        
        wandb.save(os.path.join(path, 'history.npz'))
        # Finish wandb run
        wandb.finish()

        return self.model