import os
import yaml
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def load_content(args):
    dataset_name = os.path.basename(os.path.normpath(args.root_path))
    print(dataset_name)
    if 'ETT' in args.data:
        file = 'ETT'
    elif dataset_name == 'traffic':
        file = 'Traffic'
    elif dataset_name == 'electricity':
        file = 'ECL'
    elif dataset_name == 'weather':
        file = 'Weather'
    elif dataset_name == 'illness':
        file = 'ILI'
    else:
        file = args.data
    with open('./dataset/prompt_bank/TimeVLM/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def check_path_conflict(path):
    if os.path.exists(path):
        print(f"\n[ERROR] The experiment path '{path}' already exists. Please choose a different path or remove the existing one.")
        sys.exit(1)
        
def save_args_yaml(args, path):
    if not os.path.exists(path):
        os.makedirs(path)
        
    with open(os.path.join(path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def filter_vlm_from_state_dict(state_dict, exclude_vlm=True):
    """VLM 파라미터를 state_dict에서 제외
    
    Args:
        state_dict: 모델의 state_dict
        exclude_vlm: True일 때 VLM 관련 파라미터 제외
    
    Returns:
        filtered_state_dict: VLM이 제외된 state_dict
        excluded_keys: 제외된 키 목록
    """
    if not exclude_vlm:
        return state_dict, []
    
    # VLM 관련 키 패턴
    vlm_patterns = ['vlm_manager.model', 'vlm_model', 'vlm_manager.processor']
    
    filtered_state_dict = {}
    excluded_keys = []
    
    for key, value in state_dict.items():
        # VLM 관련 키인지 확인
        is_vlm_key = any(pattern in key for pattern in vlm_patterns)
        
        if is_vlm_key:
            excluded_keys.append(key)
        else:
            filtered_state_dict[key] = value
    
    return filtered_state_dict, excluded_keys


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None, exclude_vlm=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger
        self.exclude_vlm = exclude_vlm

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.logger == None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.logger == None:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Get state_dict and optionally exclude VLM
        actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(actual_model, torch.nn.DataParallel):
            actual_model = actual_model.module
        
        state_dict = actual_model.state_dict()
        filtered_state_dict, excluded_keys = filter_vlm_from_state_dict(state_dict, self.exclude_vlm)
        
        if excluded_keys:
            log_msg = f'Excluded {len(excluded_keys)} VLM parameters from checkpoint (saving {len(filtered_state_dict)} parameters)'
            if self.logger:
                self.logger.info(log_msg)
            else:
                print(log_msg)
        
        torch.save(filtered_state_dict, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_flop(model, x_shape):
    """
    If you want to test former's flop, you need to give default value to inputs in model.forward(), 
    the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    
    try:
        from ptflops import get_model_complexity_info    
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    except ImportError:
        print('WARNING: ptflops not installed. Skipping FLOP calculation.')