from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import os
import time
import warnings
import numpy as np
import pandas as pd
import yaml
import wandb
import sys
import logging

warnings.filterwarnings('ignore')

MODEL_DICT = {
    'PatchTST': PatchTST
}

class Exp_Finetune(Exp_Basic):
    def __init__(self, args):
        super(Exp_Finetune, self).__init__(args)
        
    def _build_model(self):
        if self.args.model not in MODEL_DICT:
            raise NotImplementedError(f"Model {self.args.model} is not implemented for finetuning.")
        
        # Build backbone model
        model = MODEL_DICT[self.args.model].Model(self.args).float()
        
        # Load pretrained weights if available
        if self.args.pretrain_path is not None:
            print(f"Loading pretrained weights from: {self.args.pretrain_path}")
            pretrained_state = torch.load(self.args.pretrain_path, map_location='cpu')
        else:
            print("\033[91mTraining from scratch (no pretrained weights)\033[0m")
            pretrained_state = None
        
        # Extract backbone weights from OilO wrapper state dict
        # Exclude prediction head parameters since pred_len may differ between pretrain and finetune
        if pretrained_state is not None:
            backbone_state = {}
            excluded_keys = []
            shape_mismatch_keys = []
            
            # Get current model state for shape comparison
            current_state = model.state_dict()
            
            for key, value in pretrained_state.items():
                # OilO wrapper structure: backbone.model.backbone.xxx
                # Target structure: model.backbone.xxx
                # So we need to remove 'backbone.model.' prefix
                if key.startswith('backbone.model.'):
                    new_key = key.replace('backbone.model.', 'model.', 1)
                    
                    # Exclude prediction head parameters (they depend on pred_len which may differ)
                    # Head parameters typically have 'head' in their name
                    if 'head' in new_key.lower():
                        excluded_keys.append(new_key)
                        continue
                    
                    # Exclude RevIN parameters (trained without denormalization in pretraining)
                    # RevIN should be re-initialized for proper normalization/denormalization
                    if 'revin' in new_key.lower():
                        excluded_keys.append(new_key)
                        continue
                    
                    # Check shape compatibility
                    if new_key in current_state:
                        if current_state[new_key].shape != value.shape:
                            shape_mismatch_keys.append(
                                f"{new_key}: pretrain {value.shape} vs finetune {current_state[new_key].shape}"
                            )
                            continue  # Skip this parameter
                    
                    # Load encoder/backbone parameters only
                    backbone_state[new_key] = value
            
            if excluded_keys:
                print(f"\n[INFO] Excluded {len(excluded_keys)} prediction head parameters (will be randomly initialized):")
                for key in excluded_keys[:5]:  # Show first 5
                    print(f"  - {key}")
                if len(excluded_keys) > 5:
                    print(f"  ... and {len(excluded_keys) - 5} more")
            
            if shape_mismatch_keys:
                print(f"\n[WARNING] Shape mismatch detected - skipping {len(shape_mismatch_keys)} parameters:")
                for info in shape_mismatch_keys[:10]:  # Show first 10
                    print(f"  - {info}")
                if len(shape_mismatch_keys) > 10:
                    print(f"  ... and {len(shape_mismatch_keys) - 10} more")
                print("\n💡 Hint: This usually happens when pretrain and finetune use different seq_len.")
                print("   These parameters will be randomly initialized instead of loaded from checkpoint.")
            
            # Load backbone weights and log detailed information
            load_result = model.load_state_dict(backbone_state, strict=False)
            
            print(f"Loaded {len(backbone_state)} parameters from pretrained model")
            
            # Log unused keys from checkpoint
            if load_result.missing_keys:
                print(f"\n[INFO] Modules in model but NOT in checkpoint ({len(load_result.missing_keys)} keys):")
                for key in load_result.missing_keys:
                    print(f"  - {key}")
            
            # Log unexpected keys from checkpoint
            if load_result.unexpected_keys:
                print(f"\n[WARNING] Modules in checkpoint but NOT used in model ({len(load_result.unexpected_keys)} keys):")
                for key in load_result.unexpected_keys:
                    print(f"  - {key}")
            
            if not load_result.missing_keys and not load_result.unexpected_keys:
                print("[SUCCESS] All checkpoint parameters loaded successfully!")
        
        # Freeze backbone if linear probing mode
        if hasattr(self.args, 'linear_probing') and self.args.linear_probing:
            print("\n[Linear Probing Mode] Freezing backbone parameters...")
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                if 'head' not in name:  # Only train the head
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    trainable_count += 1
            print(f"Frozen parameters: {frozen_count}")
            print(f"Trainable parameters (head only): {trainable_count}")
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        # Only optimize parameters that require gradients (efficient for linear probing)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # PatchTST forward
                outputs = self.model(batch_x)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        save_base_path = os.path.join(self.args.finetune_result_path, self.args.model, self.args.data, setting, str(self.args.random_seed))

        path = save_base_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Setup logging to file and console
        log_file = os.path.join(path, 'train.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Override any existing configuration
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging to {log_file}")
        logger.info(f"Training configuration: {vars(self.args)}")
        
        # Save args to yaml
        with open(os.path.join(path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, indent=4)
        logger.info(f'Args saved to {path}args.yaml')

        # Initialize wandb
        mode = self.args.finetune_result_path.split('/')[-1]  # 'linear-probing', 'full-finetune', or 'scratch'
        project_name = f"OilO-{mode.title().replace('-', '')}"  # e.g., OilO-LinearProbing, OilO-FullFinetune, OilO-Scratch
        wandb.init(
            project=project_name,
            name=setting,
            tags=[self.args.des] if self.args.des else [],
            config=vars(self.args)
        )
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # Learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # PatchTST forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    # Log with global step for proper chronological ordering
                    global_step = epoch * train_steps + i
                    wandb.log({
                        'train/step_loss': loss.item(),
                        'train/learning_rate': model_optim.param_groups[0]['lr'],
                        'global_step': global_step
                    }, step=global_step)
                    
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                # Step scheduler if using TST
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            wandb.log({
                'epoch': epoch + 1,
                'train/epoch_loss': train_loss,
                'val/epoch_loss': vali_loss,
                'test/epoch_loss': test_loss
            })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        wandb.finish()

        return self.model

    def test(self, setting, test=0, save_npy=False):
        test_data, test_loader = self._get_data(flag='test')
        
        save_base_path = os.path.join(self.args.finetune_result_path, self.args.model, self.args.data, setting, str(self.args.random_seed))
        
        if test:
            print('loading model')
            # checkpoint_path = save_base_path + 'checkpoint.pth'
            checkpoint_path = os.path.join(save_base_path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))

        preds = []
        trues = []
        inputx = []
        
        # Test results folder for visualizations
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # PatchTST forward
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                # Visualize every 20 iterations
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pred_vis = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pred_vis, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print('test shape:', preds.shape, trues.shape)

        # Result save - use save_base_path
        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, mape:{}'.format(mse, mae, rse, mape))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'mse': [mse],
            'mae': [mae],
            'rse': [rse],
            'mape': [mape]
        })
        metrics_df.to_csv(os.path.join(save_base_path, 'metrics.csv'), index=False)
        print(f'Metrics saved to {os.path.join(save_base_path, "metrics.csv")}')
        
        # Optionally save predictions and ground truth as npy
        if save_npy:
            np.save(os.path.join(save_base_path, 'pred.npy'), preds)
            np.save(os.path.join(save_base_path, 'true.npy'), trues)
            print(f'Predictions saved to {os.path.join(save_base_path, "pred.npy")} and {os.path.join(save_base_path, "true.npy")}')

        return
