import argparse
import os
import yaml
from utils.tools import load_content

def main():
    parser = argparse.ArgumentParser(description='OilO Fine-tuning Script')

    # Basic Config
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    parser.add_argument('--model', type=str, required=True, default='PatchTST', help='model name')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--des', type=str, default='test', help='description of the experiment')
    
    # Pretrained Model Config
    parser.add_argument('--pretrain_base_path', type=str, default='./results/pretrain/', help='base path to pretrained model checkpoints')
    parser.add_argument('--pretrain_exp', type=str, default=None, help='pretrained experiment name (e.g., OilO_ETTh1_sl336_ll168_pl96_dm16_nh4_el3_df128_fc128_ebtimeF_itr0). If None, train from scratch.')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model checkpoint (backward compatibility)')
    parser.add_argument('--linear_probing', action='store_true', help='linear probing: freeze backbone and only train prediction head')
    parser.add_argument('--fewshot', type=int, default=100, help='number of samples for few-shot learning (%). 100: 100% data(scratch or full finetune), 10: 10% data, 1: 1% data')

    # Forecasting Task Config
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (number of variables)')
    
    # Data Config
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    
    # Model Architecture (PatchTST)
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fc dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    
    # PatchTST Backbone Config
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='percentage of cycle spent increasing learning rate (for OneCycleLR)')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pretrain_data', type=str, default=None, help='pretraining data type if different from fine-tuning data')
    parser.add_argument('--is_scratch', action='store_true', help='train from scratch without loading pretrained weights', default=False)
    parser.add_argument('--finetune_result_path', type=str, default='./results/full-finetune/', help='path to save finetune results')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--gpu_preload', action='store_true', help='preload data to gpu', default=False)
    parser.add_argument('--gpu_name', type=str, default='N/A', help='name of gpu device')

    # iTransformer
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--use_norm', type=int, default=1, help='use normalization like Non-stationary Transformer; True 1 False 0')
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function')
    
    # TimeVLM
    parser.add_argument('--use_mem_gate', action='store_true')
    parser.add_argument('--periodicity', type=int)
    parser.add_argument('--vlm_type', type=str, default='CLIP')
    parser.add_argument('--finetune_vlm', action='store_true', default=False)
    parser.add_argument('--patch_memory_size', default=100, type=int)
    parser.add_argument('--padding', default=8, type=int)
    parser.add_argument('--three_channel_image', action='store_true')
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--interpolation', type=str, default='bilinear')
    parser.add_argument('--norm_const', type=float, default=0.4)
    parser.add_argument('--learnable_image', action='store_true')
    parser.add_argument('--use_cross_attention', action='store_true', help='use cross attention to fuse image and text embeddings in customVLM')
    parser.add_argument('--save_images', action='store_true', help='save images')
    parser.add_argument('--w_out_visual', action='store_true', help='without visual part')
    parser.add_argument('--w_out_text', action='store_true', help='without text part')
    parser.add_argument('--w_out_query', action='store_true', help='without query part')
    parser.add_argument('--visualize_embeddings', action='store_true', help='visualize embeddings')
    parser.add_argument('--content', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile to speed up training (may cause issues on some environments)', default=False)
    parser.add_argument('--pre_compile', action='store_true', help='use torch.compile to speed up training (may cause issues on some environments)', default=False)
    parser.add_argument('--torch_compile_dynamic', action='store_true')

    args = parser.parse_args()

    if not args.is_scratch:
        if not args.linear_probing:
            if not args.fewshot == 100:
                # IO 병목 해소를 위해 사전 중복 체크
                setting = '{}_ft_{}_sl{}_ll{}_pl{}_{}'.format(
                    args.model,
                    args.data,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.des
                )
                
                # 중복 실험 체크
                result_path = os.path.join(args.finetune_result_path, args.model, args.data, setting, str(args.random_seed))
                args_yaml_path = os.path.join(result_path, 'args.yaml')
                metrics_csv_path = os.path.join(result_path, 'metrics.csv')
                
                if os.path.exists(args_yaml_path) and os.path.exists(metrics_csv_path):
                    print('='*80)
                    print(f'WARNING: Experiment already exists at: {result_path}')
                    print(f'  - args.yaml: Found')
                    print(f'  - metrics.csv: Found')
                    print('='*80)
                    print('Experiment already completed. Exiting to prevent duplicate runs.')
                    return
                else:
                    if not os.path.exists(args_yaml_path):
                        print(f'No existing args.yaml found at: {args_yaml_path}. Proceeding with the experiment.')
                    if not os.path.exists(metrics_csv_path):
                        print(f'No existing metrics.csv found at: {metrics_csv_path}. Proceeding with the experiment.')
                    print('='*80)
                    print('Starting new experiment...')
        # Load pretraining args to compare pretraining and finetuning data
        pretrain_args_path_candidates = []
        for data_path in os.listdir(os.path.join(args.pretrain_base_path, args.model)):
            for exp_name in os.listdir(os.path.join(args.pretrain_base_path, args.model, data_path)):
                if exp_name == args.pretrain_exp:
                    pretrain_args_path_candidates.append(os.path.join(args.pretrain_base_path, args.model, data_path, exp_name, 'args.yaml'))
        if len(pretrain_args_path_candidates) == 0:
            raise FileNotFoundError(f"Pretrained args.yaml not found for experiment: {args.pretrain_exp}")
        elif len(pretrain_args_path_candidates) > 1:
            raise ValueError(f"Multiple args.yaml found for experiment: {args.pretrain_exp}. Please specify pretrain_data explicitly.")
        pretrain_args_path = pretrain_args_path_candidates[0]

        with open(pretrain_args_path, 'r') as f:
            pretrain_args = yaml.safe_load(f)
        pretrain_args = argparse.Namespace(**pretrain_args)
        setattr(args, 'pretrain_data', pretrain_args.data)
        
        if args.pretrain_data != args.data:
            print('\033[93m' + '='*80)
            print('WARNING: Pretraining data and finetuning data are different!')
            print(f'  - Pretraining data: {pretrain_args.data}')
            print(f'  - Finetuning data: {args.data}')
            print('='*80 + '\033[0m')

    import torch
    from exp.exp_finetune import Exp_Finetune
    import random
    import numpy as np
    
    
    if args.model == 'TimeVLM':
        setattr(args, 'content', load_content(args))
    
    gpu_name = torch.cuda.get_device_name(args.gpu) if args.use_gpu else 'N/A'
    setattr(args, 'gpu_name', gpu_name)

    # Build pretrain_path from pretrain_exp if not provided
    if args.pretrain_exp is not None and args.pretrain_path is None:
        args.pretrain_path = os.path.join(args.pretrain_base_path, args.model, args.pretrain_data, args.pretrain_exp, 'checkpoint.pth')
    
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.backends.cudnn.benchmark = True

    # 병렬 실험 시 CPU 스레드 경합 방지
    # 각 프로세스가 전체 32코어를 점유하지 않도록 제한
    _num_threads = max(1, 32 // max(args.num_workers + 1, 1) // 2)
    torch.set_num_threads(_num_threads)
    os.environ['OMP_NUM_THREADS'] = str(_num_threads)
    os.environ['MKL_NUM_THREADS'] = str(_num_threads)

    # Set random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # GPU 설정
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Check if training from scratch or finetuning
    
    if args.is_scratch:
        # ANSI color codes: \033[91m = bright red, \033[0m = reset
        print('\033[91m' + '='*80)
        print('WARNING: Training from SCRATCH (no pretrained weights will be loaded)')
        print('='*80 + '\033[0m')
        setattr(args, 'pretrain_path', None)
        setattr(args, 'linear_probing', False)
        setattr(args, 'finetune_result_path', args.finetune_result_path.replace('full-finetune', 'scratch'))
        setattr(args, 'fewshot', 100)
    else:
        # Pretrained checkpoint 경로 확인
        if not os.path.exists(args.pretrain_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {args.pretrain_path}")
        print(f'Loading pretrained model from: {args.pretrain_path}')
        # mode = 'linear-probing' if args.linear_probing else 'full-finetune'
        if args.linear_probing:
            print('\033[92m' + '='*80)
            print('INFO: Using LINEAR PROBING. Change the --finetune_result_path to <path>/linear_probing')
            print('='*80 + '\033[0m')
            setattr(args, 'finetune_result_path', args.finetune_result_path.replace('finetune', 'linear_probing'))
            setattr(args, 'fewshot', 100)
        elif args.fewshot < 100:
            print('\033[94m' + '='*80)
            print(f'INFO: Using FEW-SHOT FINETUNING with {args.fewshot}% of the data.')
            print('Change the --finetune_result_path to <path>/fewshot_finetune')
            print('='*80 + '\033[0m')
            if args.fewshot == 10:
                fewshot_str = '0.10'
            elif args.fewshot == 5:
                fewshot_str = '0.05'
            elif args.fewshot == 1:
                fewshot_str = '0.01'
            else:
                raise ValueError('fewshot must be one of [100, 10, 5, 1]')
            
            setattr(args, 'finetune_result_path', args.finetune_result_path.replace('full-finetune', f'fewshot-finetune_{fewshot_str}'))

    # 실험 세팅 이름 생성
    # finetune_result_path가 변경된 후 다시 중복 체크
    setting = '{}_ft_{}_sl{}_ll{}_pl{}_{}'.format(
        args.model,
        args.data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.des
    )
    
    # 중복 실험 체크
    result_path = os.path.join(args.finetune_result_path, args.model, args.data, setting, str(args.random_seed))
    args_yaml_path = os.path.join(result_path, 'args.yaml')
    metrics_csv_path = os.path.join(result_path, 'metrics.csv')
    
    if os.path.exists(args_yaml_path) and os.path.exists(metrics_csv_path):
        print('='*80)
        print(f'WARNING: Experiment already exists at: {result_path}')
        print(f'  - args.yaml: Found')
        print(f'  - metrics.csv: Found')
        print('='*80)
        print('Experiment already completed. Exiting to prevent duplicate runs.')
        return
    else:
        if not os.path.exists(args_yaml_path):
            print(f'No existing args.yaml found at: {args_yaml_path}. Proceeding with the experiment.')
        if not os.path.exists(metrics_csv_path):
            print(f'No existing metrics.csv found at: {metrics_csv_path}. Proceeding with the experiment.')
        print('='*80)
        print('Starting new experiment...')

    # 실험 실행
    exp = Exp_Finetune(args, setting=setting)
    
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    # torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
