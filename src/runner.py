from darts.dataprocessing.transformers import Scaler
from darts.models import (
    DLinearModel, NHiTSModel, NBEATSModel, NLinearModel, TCNModel,
    TSMixerModel, TiDEModel, BlockRNNModel
)
from darts.timeseries import TimeSeries

import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='b'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_pickle(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('dt')
        df_raw = df_raw[['dt'] + cols + [self.target]]

        # print(cols)
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['dt']][border1:border2]
        df_stamp['dt'] = pd.to_datetime(df_stamp.dt)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.dt.astype(object).apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.dt.astype(object).apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.dt.astype(object).apply(lambda row: row.weekday())
            # df_stamp['hour'] = df_stamp.dt.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop('dt',axis=1).values
        elif self.timeenc == 1:
            None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
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
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(params_dict: dict, flag: str):
    data_dict = {
        'custom': Dataset_Custom,
    }

    Data = data_dict['custom']
    timeenc = 0 if params_dict.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1
        freq = params_dict.freq
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = params_dict.batch_size
        freq = params_dict.freq

    data_set = Data(
        root_path=params_dict.root_path,
        data_path=params_dict.data_path,
        flag=flag,
        size=[params_dict.seq_len, params_dict.label_len, params_dict.pred_len],
        features=params_dict.features,
        target=params_dict.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=params_dict.num_workers,
        drop_last=drop_last)

    return data_set, data_loader

class ModelFactory:
    @staticmethod
    def create_model(model_name: str, input_chunk_length: int, output_chunk_length: int, model_params: dict = None):
        if model_params is None:
            model_params = {}

        if model_name == 'dlinear':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "kernel_size": 5,
                "random_state": 42,
                "n_epochs": 50,
            }
            default_params.update(model_params)
            return DLinearModel(**default_params)

        elif model_name == 'nhits':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "num_stacks": 3,
                "num_blocks": 2,
                "num_layers": 3,
                "layer_widths": 256,
                "dropout": 0.3
            }
            default_params.update(model_params)
            return NHiTSModel(**default_params)

        elif model_name == 'nbeats':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "num_stacks": 4,
                "num_blocks": 2,
                "num_layers": 3,
                "layer_widths": 256
            }
            default_params.update(model_params)
            return NBEATSModel(**default_params)

        elif model_name == 'nlinear':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
            }
            default_params.update(model_params)
            return NLinearModel(**default_params)

        elif model_name == 'tcn':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "dropout": 0.3,
                "dilation_base": 3,
                "weight_norm": True,
                "kernel_size": 3,
                "num_filters": 32
            }
            default_params.update(model_params)
            return TCNModel(**default_params)

        elif model_name == 'tsmixer':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "hidden_size": 64,
                "ff_size": 64,
                "num_blocks": 2,
                "activation": 'ReLU',
                "dropout": 0.2,
                "norm_type": 'LayerNorm'
            }
            default_params.update(model_params)
            return TSMixerModel(**default_params)

        elif model_name == 'tide':
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "decoder_output_dim": 32,
                "hidden_size": 128
            }
            default_params.update(model_params)
            return TiDEModel(**default_params)

        elif model_name in ['rnn', 'gru', 'lstm']:
            rnn_type = model_name.upper()
            default_params = {
                "input_chunk_length": input_chunk_length,
                "output_chunk_length": output_chunk_length,
                "model": rnn_type,
                "hidden_dim": 50,
                "n_rnn_layers": 5
            }
            default_params.update(model_params)
            return BlockRNNModel(**default_params)

        else:
            raise ValueError(f"Unknown model name: {model_name}")


# --- Experiment Runner ---
class ExperimentRunner:
    def __init__(self, data: pd.DataFrame, ext_data: pd.DataFrame, target_column:str, snapshot_dt: pd.Timestamp, offset: int, horizon: int,
                 mode: str = 'valid', input_chunk_length: int = 12, output_chunk_length: int = 6,
                 output_folder: str = './results', freq: str = 'MS', parameter_dict: dict = None):
        self.data = data
        self.ext_data = ext_data
        self.target_col = target_column
        self.snapshot_dt = snapshot_dt
        self.offset = offset
        self.horizon = horizon
        self.mode = mode
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.output_folder = output_folder
        self.freq = freq
        self.scaler = Scaler()
        self.parameter_dict = parameter_dict

    def run_model(self, model_name: str, model_params: dict = None):
        target_col = self.target_col
        data = self.data
        os.makedirs(self.output_folder, exist_ok=True)
        predictions = []
        predictions_df = pd.DataFrame()
        output_file = ' '

        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')


        # TimeSeries 객체 생성 및 스케일링
        # train_series = TimeSeries.from_dataframe(train, time_col='dt', value_cols='v', freq=self.freq)
        # test_series = TimeSeries.from_dataframe(test, time_col='dt', value_cols='v', freq=self.freq)
        # train_series = TimeSeries.from_dataframe(train, time_col='dt', freq=self.freq)
        # test_series = TimeSeries.from_dataframe(test, time_col='dt', freq=self.freq)


        model = ModelFactory.create_model(model_name, self.input_chunk_length, self.output_chunk_length,
                                          model_params)
        # model.fit(train_series)
        # prediction_series = model.predict(len(test_series))
        # prediction_series = self.scaler.inverse_transform(prediction_series)
        #
        # for i, dt in enumerate(test_series.time_index):
        #     record = {'grain_id': grain_id, 'dt': dt, 'pred': prediction_series.values()[i][0]}
        #     if self.mode == 'valid':
        #         record['v'] = test_series.values()[i][0]
        #         record['model'] = model_name
        #     else:
        #         record['snapshot_dt'] = self.snapshot_dt
        #     predictions.append(record)
        #
        # predictions_df = pd.DataFrame(predictions)
        #
        #
        # if self.mode == 'valid':
        #     output_file = os.path.join(self.output_folder, f'predictions_monthly_{model_name}_valid_v3.csv')
        # elif self.mode == 'test':
        #     output_file = os.path.join(self.output_folder, f'predictions_monthly_{model_name}_next_v3.csv')
        # else:
        #     output_file = os.path.join(self.output_folder, f'predictions_monthly_{model_name}_v3.csv')
        return predictions_df, output_file
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
