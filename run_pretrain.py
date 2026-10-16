import argparse
import os
import torch
from exp.exp_pretrain import Exp_Pretrain
import random
import numpy as np
import yaml

def main():
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='OilO Pre-training Script')

    # Basic Config
    parser.add_argument('--model', type=str, required=True, default='PatchTST', help='model name')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain_results_path', type=str, default='results/pretrain', help='path to save pretrain results')
    parser.add_argument('--des', type=str, default='default_description', help='description of the experiment')

    # Forecasting Task Config (Backbone: PatchTST)
    parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length') # Pretrain에선 안 쓰이지만 Backbone 초기화 호환성 위해 필요
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (number of variables)')
    
    # Data Config
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    
    # Model Architecture (PatchTST & OilO)
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fc dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--module_version', type=str, default='v1-triangular-causal', help='module version')
    parser.add_argument('--fast_ci_olio', action='store_true', help='call oilo_v1_triangular_causal_channel_individual_fast module')
    parser.add_argument('--fast_ci_oilo', action='store_true', help='call oilo_v1_triangular_causal_channel_individual_fast module')
    
    # PatchTST Backbone Config
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--padding_patch', type=str, default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    
    # OilO Specific Params
    parser.add_argument('--projection_dim', type=int, default=128, help='dimension of projection head output')
    parser.add_argument('--max_tau', type=int, default=48, help='maximum time shift (tau) for contrastive learning')
    # parser.add_argument('--lambda_std', type=float, default=0.01, help='weight for variance regularization') # 삭제됨 (Contrastive만 사용)
    parser.add_argument('--contrastive_only', action='store_true', help='use only contrastive loss for pretraining')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate') # exp_basic에서 사용될 수 있음
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--gpu_preload', action='store_true', help='preload data to gpu', default=False)
    parser.add_argument('--gpu_name', type=str, default='N/A', help='name of gpu device')
    parser.add_argument('--use_amp', action='store_true')

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
    
    # SoftCLT Specific Params
    parser.add_argument('--softclt_tau', type=float, default=5, help='SoftCLT temperature parameter')
    parser.add_argument('--softclt_alpha', type=float, default=1.0, help='SoftCLT alpha scaling parameter')

    # TimeSiam Specific Params
    parser.add_argument('--timesiam_num_lineages', type=int, default=6, help='number of lineage embeddings in TimeSiam')
    parser.add_argument('--torch_compile', action='store_true', help='use torch.compile to speed up training (may cause issues on some environments)', default=False)
    parser.add_argument('--fewshot', type=int, default=100, help='It should be 100 in pretrain')

    args = parser.parse_args()
    
    args.fewshot = 100
    
    if args.fast_ci_olio == True or args.fast_ci_oilo == True:
        args.fast_ci_olio = True
        args.fast_ci_oilo = True

    gpu_name = torch.cuda.get_device_name(args.gpu) if args.use_gpu else 'N/A'
    setattr(args, 'gpu_name', gpu_name)

    # GPU 설정
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    torch.backends.cudnn.benchmark = True
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.contrastive_only:
        setattr(args, 'max_tau', -1)

    # 실험 세팅 이름 생성 (결과 저장 폴더명 구분용)
    # 예: OilO_ETTh1_sl512_pl12_tau48_128_default_description
    if '_' in args.des:
        assert ValueError("Underscores are not allowed in the description (des) argument.")
    
    setting = '{}_{}_sl{}_pl{}_tau{}_{}_{}'.format(
        args.model,
        args.data,
        args.seq_len,
        args.patch_len,
        args.max_tau,
        args.projection_dim,
        args.des
    ) # 0은 실험 횟수나 식별자

    if args.module_version != 'v1-triangular-causal':
        setattr(args, 'pretrain_results_path', 
                f'{args.pretrain_results_path}_{args.module_version}')


    print('Args in experiment:')
    print(args)
    
    result_path = os.path.join(args.pretrain_results_path, args.model, args.data, setting)
    args_path = os.path.join(result_path, 'args.yaml')
    checkpoint_path = os.path.join(result_path, 'checkpoint.pth')
    history_path = os.path.join(result_path, 'history.npz')
    finish_path = os.path.join(result_path, '.finish')
    
    print('Save path: ', result_path)
    
    check_list = [checkpoint_path, history_path, finish_path]

    if all(os.path.exists(path) for path in check_list):
        print('='*80)
        print(f'WARNING: Experiment already exists at: {result_path}')
        print(f'  - args.yaml: {"Found" if os.path.exists(args_path) else "Not Found"}')
        print(f'  - checkpoint.pth: Found')
        print(f'  - history.npz: Found')
        print(f'  - .finish: Found')
        print('='*80)
        
        if not os.path.exists(args_path):
            print('Saving args.yaml...')
            os.makedirs(result_path, exist_ok=True)
            with open(os.path.join(result_path, 'args.yaml'), 'w') as f:
                yaml.dump(vars(args), f, indent=4)
            
        return
    
    # 실험 실행
    exp = Exp_Pretrain(args)
    
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    try:
        exp.train(setting)
    except torch.cuda.OutOfMemoryError as e:
        with open(os.path.join(result_path, 'oom_error.log'), 'w') as f:
            f.write('OOM Error during training.\n')
            f.write(str(e))
        print(f'❌ OOM Error during training. Logged to {os.path.join(result_path, "oom_error.log")}')
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()