from data_provider.data_loader import LTSFDataset
from torch.utils.data import DataLoader
import torch

class GPUPreloadDataLoader:
    def __init__(self, dataset, dataloader, device="cuda"):
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        
        self.data_x = self.dataset.data_x.to(device)
        self.data_y = self.dataset.data_y.to(device)
        self.data_stamp = self.dataset.data_stamp.to(device)
        
        self.flag = self.dataset.flag
        
        if self.flag == 'pretrain':
            self.total_window = self.dataset.total_window
        else:
            self.seq_len = self.dataset.seq_len
            self.label_len = self.dataset.label_len
            self.pred_len = self.dataset.pred_len
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for idx in self.dataloader:
            idx = idx.to(self.device)
            
            if self.flag == 'pretrain':
                offsets = torch.arange(self.total_window, device=self.device)
                indices = idx.unsqueeze(1) + offsets
                
                seq_long = self.data_x[indices]
                seq_mark_long = self.data_stamp[indices]
                
                yield seq_long, seq_mark_long
            else:
                x_offsets = torch.arange(self.seq_len, device=self.device)
                y_offsets = torch.arange(self.seq_len - self.label_len, 
                                         self.seq_len + self.pred_len, device=self.device)
                
                # 2. 브로드캐스팅으로 인덱스 매트릭스 생성
                x_indices = idx.unsqueeze(1) + x_offsets
                y_indices = idx.unsqueeze(1) + y_offsets
                
                # 3. 한 번에 슬라이싱
                seq_x = self.data_x[x_indices]
                seq_y = self.data_y[y_indices]
                seq_x_mark = self.data_stamp[x_indices]
                seq_y_mark = self.data_stamp[y_indices]

                yield seq_x, seq_y, seq_x_mark, seq_y_mark

def data_provider(args, flag, device, shuffle=None):
    # args.data (예: 'ETTh1')를 그대로 dataset 인자로 넘김
    timeenc = 0 if args.embed != 'timeF' else 1

    fewshot = getattr(args, 'fewshot', 100)

    if flag != 'pretrain':
        if fewshot != 100:
            args.gpu_preload = False

    if flag == 'test':
        shuffle_flag = False if shuffle is None else shuffle
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pretrain':
        shuffle_flag = True if shuffle is None else shuffle
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else: # train, val
        shuffle_flag = True if shuffle is None else shuffle
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    gpu_preload = args.gpu_preload if hasattr(args, 'gpu_preload') else False
    if gpu_preload:
        print("\033[93m" + "Warning: Using GPU preload may lead to out-of-memory (OOM) errors if the dataset is too large to fit in GPU memory." + "\033[0m")
        print("Using GPU preload for data...")
        print(f"Device: {device}")
        print(f"The num_workers argument becomes {args.num_workers} -> 0 because of GPU preload.")
        num_workers = 0
    else:
        # Multi-process data loading is fully compatible with GPU training
        num_workers = args.num_workers
    

    data_set = LTSFDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        dataset=args.data,     # 'ETTh1', 'custom' 등 문자열 전달
        max_tau=args.max_tau if hasattr(args, 'max_tau') else 24,
        gpu_preload=gpu_preload,
        device=device
    )

    if flag != 'pretrain':
        try:
            if fewshot != 100 and flag == 'train':
                assert fewshot < 100
                original_len = len(data_set)
                num_samples = int(original_len * float(fewshot) / 100.0)
                indices = torch.randperm(original_len)[:num_samples]
                data_set = torch.utils.data.Subset(data_set, indices)
                print(f"Using {fewshot}% of the data: {num_samples} samples. (Actual: {num_samples} / {original_len})")
                drop_last = False
        except Exception as e:
            print(f"Error during fewshot subsetting: {e}")
            raise
    # Always use custom collate function for consistent behavior
    # This preserves device (GPU/CPU) of input tensors for reproducibility
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False if gpu_preload else True,
        persistent_workers=False if num_workers == 0 else True,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    if gpu_preload:
        data_loader = GPUPreloadDataLoader(data_set, data_loader, device=device)
    
    return data_set, data_loader