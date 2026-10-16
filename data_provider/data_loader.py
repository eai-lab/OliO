import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class LTSFDataset(Dataset):
    def __init__(
        self, root_path, flag='train', size=None,
        features='S', data_path=None,
        target='OT', scale=True, timeenc=0, freq='h',
        dataset=None, max_tau=None, gpu_preload=False, device='cuda'
    ):
        self.gpu_preload = gpu_preload
        self.device = device
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val', 'pretrain']
        
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pretrain': 0}
        self.set_type = type_map[flag]

        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # OilO: Pre-training을 위한 설정
        self.max_tau = max_tau if max_tau is not None else 24
        self.total_window = self.seq_len + self.max_tau
        
        # 데이터셋 종류 식별 (소문자로 통일)
        self.dataset = dataset.lower() if dataset else 'custom'

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # 1. 데이터셋 종류에 따른 Border(데이터 분할 기준) 설정
        if self.dataset in ['etth1', 'etth2']:
            # ETT Hour: 12개월 / 4개월 / 4개월
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        
        elif self.dataset in ['ettm1', 'ettm2']:
            # ETT Minute: 12개월 / 4개월 / 4개월 (분 단위 데이터 양 고려)
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        
        else: # Custom Dataset
            # Custom: 70% / 10% / 20% 분할
            # 컬럼 정리 (target을 맨 뒤로)
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 2. Features(변수) 선택 ('M', 'MS', 'S')
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # date 제외한 모든 컬럼
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]] # target 컬럼만

        # 3. Scaling (Train 데이터 기준으로 fit)
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 4. Time Encoding (시간 특징 추출)
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 수동 인코딩
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            
            # 분 단위 데이터셋인 경우 'minute' 추가
            if self.dataset in ['ettm1', 'ettm2'] or self.freq == 't':
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15) # 15분 단위
            
            data_stamp = df_stamp.drop(['date'], axis=1).values
            
        elif self.timeenc == 1:
            # time_features 라이브러리 사용
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        self.data_x = torch.tensor(self.data_x, dtype=torch.float32)  # numpy -> CPU tensor
        self.data_y = torch.tensor(self.data_y, dtype=torch.float32)  # numpy -> CPU tensor
        self.data_stamp = torch.tensor(self.data_stamp, dtype=torch.float32)  # numpy -> CPU tensor

    def __getitem__(self, index):
        # Case A: Pre-training (Long Sequence with Overlap)
        if self.gpu_preload:
            return index
        
        if self.flag == 'pretrain':
            s_begin = index
            s_end = s_begin + self.total_window
            
            # 범위 체크 및 보정
            if s_end > len(self.data_x):
                s_end = len(self.data_x)
                s_begin = s_end - self.total_window
            
            seq_long = self.data_x[s_begin:s_end]
            seq_mark_long = self.data_stamp[s_begin:s_end]
            
            return seq_long, seq_mark_long

        # Case B: Standard Forecasting (Input Sequence, Label)
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        if self.flag == 'pretrain':
            return len(self.data_x) - self.total_window + 1
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)