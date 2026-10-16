from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchTST, iTransformer, TimeVLM
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
# import wandb
import sys
import logging
from modules import get_wrapper
from contextlib import nullcontext

warnings.filterwarnings('ignore')

        
MODEL_DICT = {
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'TimeVLM': TimeVLM
}

class Exp_Finetune(Exp_Basic):
    def __init__(self, args, setting=None):
        self.args = args
        self.logger = None  # Will be initialized in train()
        path = os.path.join(self.args.finetune_result_path, self.args.model, self.args.data, setting, str(self.args.random_seed))
        if not os.path.exists(path):
            os.makedirs(path)
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
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to {log_file}")
        self.logger.info(f"Finetuning configuration: {vars(self.args)}")
        self.is_timesiam = 'timesiam' in self.args.pretrain_path.lower() if self.args.pretrain_path else False
        
        if self.args.use_amp:
            self.logger.info("AMP is enabled for finetuning.")
        
        super(Exp_Finetune, self).__init__(args)

                
    def _build_model(self):
        if self.args.model not in MODEL_DICT:
            raise NotImplementedError(f"Model {self.args.model} is not implemented for finetuning.")
        
        # Build backbone model
        backbone = MODEL_DICT[self.args.model].Model(self.args).float()
        if self.is_timesiam:
            self.logger.info("Detected TimeSiam - using TimeSiam wrapper in Finetune.")
            # insert 'module_version' argument for TimeSiam wrapper. Since there is no module_version in args, we add it here.
            setattr(self.args, 'module_version', 'timesiam')
            setattr(self.args, 'max_tau', self.args.seq_len)
            setattr(self.args, 'timesiam_num_lineages', 6)
            wrapper = get_wrapper(self.args, backbone).float()
            model = wrapper
        else:
            model = backbone
      
        # Load pretrained weights if available
        if self.args.pretrain_path is not None:
            self.logger.info(f"Loading pretrained weights from: {self.args.pretrain_path}")
            pretrained_state = torch.load(self.args.pretrain_path, map_location='cpu')
        else:
            self.logger.warning("Training from scratch (no pretrained weights)")
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
                if self.is_timesiam:
                    exclude_keywords = ['projector', 'decoder', 'revin']
                    if any(ex_key in key.lower() for ex_key in exclude_keywords):
                        excluded_keys.append(key)
                        continue
                    
                    if key in current_state:
                        if current_state[key].shape != value.shape:
                            shape_mismatch_keys.append(
                                f"{key}: pretrain {value.shape} vs finetune {current_state[key].shape}"
                            )
                            continue  # Skip this parameter
                        backbone_state[key] = value
                else:
                    if key.startswith('backbone.'):
                        if key.startswith('backbone.model.'):
                            # PatchTST의 경우, backbone.model.backbone으로 돼 있기에 이런 안전 장치가 필요.
                            new_key = key.replace('backbone.model.', 'model.', 1)
                        else:
                            new_key = key.replace('backbone.', '', 1)
                        
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
                self.logger.info(f"Excluded {len(excluded_keys)} prediction head parameters (will be randomly initialized):")
                for key in excluded_keys[:5]:  # Show first 5
                    self.logger.info(f"  - {key}")
                if len(excluded_keys) > 5:
                    self.logger.info(f"  ... and {len(excluded_keys) - 5} more")
            
            if shape_mismatch_keys:
                self.logger.warning(f"Shape mismatch detected - skipping {len(shape_mismatch_keys)} parameters:")
                for info in shape_mismatch_keys[:10]:  # Show first 10
                    self.logger.warning(f"  - {info}")
                if len(shape_mismatch_keys) > 10:
                    self.logger.warning(f"  ... and {len(shape_mismatch_keys) - 10} more")
                self.logger.info("💡 Hint: This usually happens when pretrain and finetune use different seq_len or d_ff.")
                self.logger.info("   These parameters will be randomly initialized instead of loaded from checkpoint.")
            
            # Load backbone weights and log detailed information
            load_result = model.load_state_dict(backbone_state, strict=False)
            
            self.logger.info(f"Loaded {len(backbone_state)} parameters from pretrained model")
            
            # Log successfully loaded weights
            if backbone_state:
                self.logger.info(f"Successfully loaded weights ({len(backbone_state)} keys):")
                for idx, key in enumerate(sorted(backbone_state.keys())):
                    self.logger.info(f"  ✓ {key}: {tuple(backbone_state[key].shape)}")
            
            # Log unused keys from checkpoint
            if load_result.missing_keys:
                self.logger.info(f"Modules in model but NOT in checkpoint ({len(load_result.missing_keys)} keys):")
                for key in load_result.missing_keys:
                    self.logger.info(f"  - {key}")
            
            # Log unexpected keys from checkpoint
            if load_result.unexpected_keys:
                self.logger.warning(f"Modules in checkpoint but NOT used in model ({len(load_result.unexpected_keys)} keys):")
                for key in load_result.unexpected_keys:
                    self.logger.warning(f"  - {key}")
            
            if not load_result.missing_keys and not load_result.unexpected_keys:
                self.logger.info("All checkpoint parameters loaded successfully!")
        
        # Freeze backbone if linear probing mode
        if hasattr(self.args, 'linear_probing') and self.args.linear_probing:
            self.logger.info("[Linear Probing Mode] Freezing backbone parameters...")
            frozen_count = 0
            trainable_count = 0
            for name, param in model.named_parameters():
                if 'head' not in name:  # Only train the head
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    trainable_count += 1
            self.logger.info(f"Frozen parameters: {frozen_count}")
            self.logger.info(f"Trainable parameters (head only): {trainable_count}")
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # if self.args.torch_compile: This is deprecated code since the on-demand compile time is longer than the actual training time for finetunening.
        #     self.logger.info("Compiling model with torch.compile...")
        #     try:
        #         model = torch.compile(model, dynamic=True)
        #         self.logger.info("Model compiled successfully!")
        #     except Exception as e:
        #         self.logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
        use_compile = getattr(self.args, 'pre_compile', False) or getattr(self.args, 'torch_compile', False)
        if use_compile:
            self.logger.info("Applying torch.compile configuration...")
            
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.abspath('./torch_compiled_cache')
            try:
                os.environ["TORCH_LOGS"] = "inductor"
                dynamic = getattr(self.args, 'torch_compile_dynamic', False)
                model = torch.compile(model, dynamic=dynamic)
                self.logger.info("Model wrapped with torch.compile successfully.")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, device=self.device)
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
        amp_context = torch.cuda.amp.autocast() if self.args.use_amp else nullcontext()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                
                if not self.args.gpu_preload:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_x_mark = batch_x_mark.to(self.device, non_blocking=True)
                    batch_y_mark = batch_y_mark.to(self.device, non_blocking=True)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # PatchTST forward
                with amp_context:
                    outputs = self.model(
                        batch_x=batch_x, 
                        batch_x_mark=batch_x_mark, 
                        dec_inp=dec_inp, 
                        batch_y_mark=batch_y_mark, 
                        return_latent=False,
                    )
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.detach())
                
        total_loss = torch.stack(total_loss).mean().item()
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        save_base_path = os.path.join(self.args.finetune_result_path, self.args.model, self.args.data, setting, str(self.args.random_seed))

        path = save_base_path
        
        self.logger.info(f"Training configuration: {vars(self.args)}")
        
        # Save args to yaml
        with open(os.path.join(path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, indent=4)
        self.logger.info(f'Args saved to {path}args.yaml')

        # Initialize wandb
        if self.args.finetune_result_path[-1] == '/':
            mode = self.args.finetune_result_path.split('/')[-2]  # 'linear-probing', 'full-finetune', or 'scratch'
        else:
            mode = self.args.finetune_result_path.split('/')[-1]  # 'linear-probing', 'full-finetune', or 'scratch'
        
        time_now = time.time()
        train_steps = len(train_loader)
        print("Setup early stopping...")
        # VLM 제외 여부 결정: finetune_vlm=False일 때만 제외
        exclude_vlm = not getattr(self.args, 'finetune_vlm', False)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger, exclude_vlm=exclude_vlm)

        print("Get optimizer and loss function")
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(f"Set Amp mode to: {self.args.use_amp}")
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        context = torch.cuda.amp.autocast() if self.args.use_amp else nullcontext()
        
        print(f"Load learning rate scheduler")
        # Learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )
        
        if getattr(self.args, 'pre_compile', False):
            self.logger.info("🔥 [Pre-compile Mode] Running a single dummy pass to build kernel cache...")
            self.model.train()
            
            # 첫 번째 배치 추출
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
            batch_x, batch_y = batch_x.float(), batch_y.float()
            batch_x_mark, batch_y_mark = batch_x_mark.float(), batch_y_mark.float()

            if not getattr(self.args, 'gpu_preload', False):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.to(self.device), batch_y_mark.to(self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
            
            with context:
                outputs = self.model(
                    batch_x=batch_x, 
                    batch_x_mark=batch_x_mark, 
                    dec_inp=dec_inp, 
                    batch_y_mark=batch_y_mark, 
                    return_latent=False
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                loss = criterion(outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:])
                
            scaler.scale(loss).backward()
            
            self.logger.info("🔥 Running a dummy validation pass to build eval cache...")
            self.model.eval()
            with torch.no_grad():
                val_x, val_y, val_x_mark, val_y_mark = next(iter(vali_loader))
                val_x, val_y = val_x.float().to(self.device), val_y.float().to(self.device)
                val_x_mark, val_y_mark = val_x_mark.float().to(self.device), val_y_mark.float().to(self.device)
                val_dec_inp = torch.zeros_like(val_y[:, -self.args.pred_len:, :])
                val_dec_inp = torch.cat([val_y[:, :self.args.label_len, :], val_dec_inp], dim=1)
                with context:
                    _ = self.model(
                        batch_x=val_x, batch_x_mark=val_x_mark, 
                        dec_inp=val_dec_inp, batch_y_mark=val_y_mark, return_latent=False
                    )
            # =======================================================
            
            self.logger.info("✅ Cache generation complete! Compiled files are saved in ./torch_compiled_cache")
            import sys
            sys.exit(0)
            

        print(f"Train begin")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f"Epoch: {epoch}")
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if i < 5 and epoch == 0:
                    print(f"Load data")
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                
                if not self.args.gpu_preload:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_x_mark = batch_x_mark.to(self.device, non_blocking=True)
                    batch_y_mark = batch_y_mark.to(self.device, non_blocking=True)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                if i < 5 and epoch == 0:
                    print(f"Forward Epoch: {epoch}, iter: {i}")

                with context:
                    outputs = self.model(
                        batch_x=batch_x, 
                        batch_x_mark=batch_x_mark, 
                        dec_inp=dec_inp, 
                        batch_y_mark=batch_y_mark, 
                        return_latent=False
                    )
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.detach())
                    
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

                if i < 5 and epoch == 0:
                    print(f"Train Epoch: {epoch}, iter: {i}, loss: {loss.item()}")


                if (i + 1) % 100 == 0:
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                                 
                    iter_count = 0
                    time_now = time.time()

                
                # Step scheduler if using TST
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(train_loss)
            train_loss = torch.stack(train_loss).mean().item()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # try:
            #     wandb.log({
            #         'epoch': epoch + 1,
            #         'train/epoch_loss': train_loss,
            #         'val/epoch_loss': vali_loss,
            #         'test/epoch_loss': test_loss
            #     })
            # except Exception as e:
            #     pass
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                self.logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, 'checkpoint.pth')
        actual_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        if isinstance(actual_model, torch.nn.DataParallel):
            actual_model = actual_model.module
        actual_model.load_state_dict(torch.load(best_model_path), strict=False)
        
        # try:
        #     wandb.finish()
        # except Exception as e:
        #     pass

        return actual_model

    def test(self, setting, test=0, save_npy=False):
        test_data, test_loader = self._get_data(flag='test')
        
        save_base_path = os.path.join(self.args.finetune_result_path, self.args.model, self.args.data, setting, str(self.args.random_seed))
        logger = logging.getLogger(__name__)

        test_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        if test:
            logger.info('Loading model from checkpoint')
            # checkpoint_path = save_base_path + 'checkpoint.pth'
            checkpoint_path = os.path.join(save_base_path, 'checkpoint.pth')
            target_model = test_model.module if isinstance(test_model, torch.nn.DataParallel) else test_model
            target_model.load_state_dict(torch.load(checkpoint_path), strict=False)

        preds = []
        trues = []
        inputx = []
        
        # Test results folder for visualizations
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_model.eval()
        amp_context = torch.cuda.amp.autocast() if self.args.use_amp else nullcontext()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()
                
                if not self.args.gpu_preload:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    batch_x_mark = batch_x_mark.to(self.device, non_blocking=True)
                    batch_y_mark = batch_y_mark.to(self.device, non_blocking=True)
                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                # PatchTST forward
                with amp_context:
                    outputs = test_model(
                        batch_x=batch_x, 
                        batch_x_mark=batch_x_mark, 
                        dec_inp=dec_inp, 
                        batch_y_mark=batch_y_mark, 
                        return_latent=False
                    )

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
        
        logger.info('test shape: {}, {}'.format(preds.shape, trues.shape))

        # Result save - use save_base_path
        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        logger.info('mse:{}, mae:{}, rse:{}, mape:{}'.format(mse, mae, rse, mape))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'mse': [mse],
            'mae': [mae],
            'rse': [rse],
            'mape': [mape]
        })
        metrics_df.to_csv(os.path.join(save_base_path, 'metrics.csv'), index=False)
        logger.info(f'Metrics saved to {os.path.join(save_base_path, "metrics.csv")}')
        
        # Optionally save predictions and ground truth as npy
        if save_npy:
            np.save(os.path.join(save_base_path, 'pred.npy'), preds)
            np.save(os.path.join(save_base_path, 'true.npy'), trues)
            logger.info(f'Predictions saved to {os.path.join(save_base_path, "pred.npy")} and {os.path.join(save_base_path, "true.npy")}')
        logger.info(f'Finish')
        return
