# part of this code are copied from DCRNN
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch_geometric.data import Batch, Data
# from sklearn.externals import joblib

class DataLoader(object):

    def __init__(self,
                 x_od,
                 y_od,
                 unfinished,
                 history,
                 yesterday,
                 x_do,
                 y_do,
                 xtime,
                 ytime,
                 batch_size,
                 pad_with_last_sample=True,
                 shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x_od) % batch_size)) % batch_size
            x_od_padding = np.repeat(x_od[-1:], num_padding, axis=0)
            x_do_padding = np.repeat(x_do[-1:], num_padding, axis=0)
            xtime_padding = np.repeat(xtime[-1:], num_padding, axis=0)
            y_od_padding = np.repeat(y_od[-1:], num_padding, axis=0)
            y_do_padding = np.repeat(y_do[-1:], num_padding, axis=0)
            ytime_padding = np.repeat(ytime[-1:], num_padding, axis=0)

            x_od = np.concatenate([x_od, x_od_padding], axis=0)
            x_do = np.concatenate([x_do, x_do_padding], axis=0)
            xtime = np.concatenate([xtime, xtime_padding], axis=0)
            y_od = np.concatenate([y_od, y_od_padding], axis=0)
            y_do = np.concatenate([y_do, y_do_padding], axis=0)
            ytime = np.concatenate([ytime, ytime_padding], axis=0)

            unfi_num_padding = (batch_size - (len(unfinished) % batch_size)) % batch_size
            unfinished_padding = np.repeat(unfinished[-1:], unfi_num_padding, axis=0)
            unfinished = np.concatenate([unfinished, unfinished_padding], axis=0)

            history_num_padding = (batch_size - (len(history) % batch_size)) % batch_size
            history_padding = np.repeat(history[-1:], history_num_padding, axis=0)
            history = np.concatenate([history, history_padding], axis=0)

            yesterday_num_padding = (batch_size - (len(yesterday) % batch_size)) % batch_size
            yesterday_padding = np.repeat(yesterday[-1:], yesterday_num_padding, axis=0)
            yesterday = np.concatenate([yesterday, yesterday_padding], axis=0)
        self.size = len(x_od)
        self.num_batch = int(self.size // self.batch_size)
        # if shuffle:
        #     permutation = np.random.permutation(self.size)
        #     x_do, y_do = x_do[permutation], y_do[permutation]
        #     x_od, y_od = x_od[permutation], y_od[permutation]
        #     xtime, ytime = xtime[permutation], ytime[permutation]
        #     unfinished = unfinished[permutation]
        #     history = history[permutation]
        #     yesterday = yesterday[permutation]
        self.x_do = x_do
        self.y_do = y_do
        self.x_od = x_od
        self.y_od = y_od
        self.unfinished = unfinished
        self.history = history
        self.yesterday = yesterday
        self.xtime = xtime
        self.ytime = ytime

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        x_do, y_do = self.x_do[permutation], self.y_do[permutation]
        x_od, y_od = self.x_od[permutation], self.y_od[permutation]
        xtime, ytime = self.xtime[permutation], self.ytime[permutation]
        unfinished = self.unfinished[permutation]
        history = self.history[permutation]
        yesterday = self.yesterday[permutation]
        self.x_do = x_do
        self.y_do = y_do
        self.x_od = x_od
        self.y_od = y_od
        self.unfinished = unfinished
        self.history = history
        self.yesterday = yesterday
        self.xtime = xtime
        self.ytime = ytime

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size,
                              self.batch_size * (self.current_ind + 1))
                x_do_i = self.x_do[start_ind:end_ind, ...]
                y_do_i = self.y_do[start_ind:end_ind, ...]
                x_od_i = self.x_od[start_ind:end_ind, ...]
                y_od_i = self.y_od[start_ind:end_ind, ...]
                unfinished_i = self.unfinished[start_ind:end_ind, ...]
                history_i = self.history[start_ind:end_ind, ...]
                yesterday_i = self.yesterday[start_ind:end_ind, ...]
                xtime_i = self.xtime[start_ind:end_ind, ...]
                ytime_i = self.ytime[start_ind:end_ind, ...]
                yield (x_od_i, y_od_i, x_do_i, y_do_i, unfinished_i, history_i, yesterday_i, xtime_i, ytime_i)  # 注意顺序
                self.current_ind += 1
        return _wrapper()


class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def load_dataset(dataset_dir,
                    do_dataset_dir,
                    batch_size,
                    test_batch_size=None,
                    scaler_axis=(0,
                                 1,
                                 2,
                                 3),
                    **kwargs):
    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = load_pickle(os.path.join(dataset_dir, category + '.pkl'))
        history_data = load_pickle(os.path.join(dataset_dir, category + '_history_long.pkl'))
        yesterday_data = load_pickle(os.path.join(dataset_dir, category + '_history_short.pkl'))
        do_data = load_pickle(os.path.join(do_dataset_dir, category + '_do.pkl'))
        data['x_' + category] = cat_data['finished']
        data['unfinished_' + category] = cat_data['unfinished']
        data['xtime_' + category] = cat_data['xtime']
        data['y_' + category] = cat_data['y']
        data['ytime_' + category] = cat_data['ytime']
        data['do_x_'+ category] = do_data['do_x']
        data['do_y_' + category] = do_data['do_y']

        his_sum = np.sum(history_data['history'], axis=-1)
        history_distribution = np.nan_to_num(np.divide(history_data['history'], np.expand_dims(his_sum, axis=-1)))
        data['history_' + category] = np.multiply(history_distribution, cat_data['unfinished'])

        yesterday_sum = np.sum(yesterday_data['history'], axis=-1)
        yesterday_distribution = np.nan_to_num(np.divide(yesterday_data['history'], np.expand_dims(yesterday_sum, axis=-1)))
        data['yesterday_' + category] = np.multiply(yesterday_distribution, cat_data['unfinished'])


    scaler = StandardScaler(mean=data['x_train'].mean(axis=scaler_axis),
                            std=data['x_train'].std(axis=scaler_axis))
    do_scaler = StandardScaler(mean=data['do_x_train'].mean(axis=scaler_axis),
                                    std=data['do_x_train'].std(axis=scaler_axis))
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
        data['do_x_' + category] = do_scaler.transform(data['do_x_' + category])
        data['do_y_' + category] = do_scaler.transform(data['do_y_' + category])
        data['unfinished_' + category] = scaler.transform(data['unfinished_' + category])
        data['history_' + category] = scaler.transform(data['history_' + category])
        data['yesterday_' + category] = scaler.transform(data['yesterday_' + category])

    data['train_loader'] = DataLoader(data['x_train'],
                                      data['y_train'],
                                      data['unfinished_train'],
                                      data['history_train'],
                                      data['yesterday_train'],
                                      data['do_x_train'],
                                      data['do_y_train'],
                                      data['xtime_train'],
                                      data['ytime_train'],
                                      batch_size,
                                      shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'],
                                    data['y_val'],
                                    data['unfinished_val'],
                                    data['history_val'],
                                    data['yesterday_val'],
                                    data['do_x_val'],
                                    data['do_y_val'],
                                    data['xtime_val'],
                                    data['ytime_val'],
                                    test_batch_size,
                                    shuffle=False)

    data['test_loader'] = DataLoader(data['x_test'],
                                     data['y_test'],
                                     data['unfinished_test'],
                                     data['history_test'],
                                     data['yesterday_test'],
                                     data['do_x_test'],
                                     data['do_y_test'],
                                     data['xtime_test'],
                                     data['ytime_test'],
                                     test_batch_size,
                                     shuffle=False)
    data['scaler'] = scaler
    data['do_scaler'] = do_scaler

    return data

def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class SimpleBatch(list):

    def to(self, device):
        for ele in self:
            ele.to(device)
        return self


def collate_wrapper(x_od, y_od, x_do, y_do, unfinished, history, yesterday, edge_index, edge_attr, seq_len, horizon, device, return_y=True):
    x_od = torch.tensor(x_od, dtype=torch.float, device=device)
    y_od = torch.tensor(y_od, dtype=torch.float, device=device)
    x_od = x_od.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_od_T_first = y_od.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)

    x_do = torch.tensor(x_do, dtype=torch.float, device=device)
    y_do = torch.tensor(y_do, dtype=torch.float, device=device)
    x_do = x_do.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)
    y_do_T_first = y_do.transpose(dim0=1, dim1=0)  # (T, N, num_nodes, num_features)

    unfinished = torch.tensor(unfinished, dtype=torch.float, device=device)
    unfinished = unfinished.transpose(dim0=1, dim1=0)

    history = torch.tensor(history, dtype=torch.float, device=device)
    history = history.transpose(dim0=1, dim1=0)

    yesterday = torch.tensor(yesterday, dtype=torch.float, device=device)
    yesterday = yesterday.transpose(dim0=1, dim1=0)
    #  do not tranpose y_truth
    T = x_od.size()[0]
    H = y_od_T_first.size()[0]
    N = x_od.size()[1]
    # generate batched sequence.
    sequences = []
    sequences_y = []
    for t in range(T - seq_len, T):
        cur_batch_x_od = x_od[t]
        cur_batch_x_do = x_do[t]
        cur_batch_unfinished = unfinished[t]
        cur_batch_history = history[t]
        cur_batch_yesterday = yesterday[t]

        batch = Batch.from_data_list([
            Data(x_do=cur_batch_x_do[i],
                 x_od=cur_batch_x_od[i],
                 unfinished=cur_batch_unfinished[i],
                 history=cur_batch_history[i],
                 yesterday=cur_batch_yesterday[i],
                 edge_index=edge_index,
                 edge_attr=edge_attr) for i in range(N)
        ])
        sequences.append(batch)

    for t in range(H - horizon, H):
        cur_batch_y_od = y_od_T_first[t]
        cur_batch_y_do = y_do_T_first[t]

        batch_y = Batch.from_data_list([
            Data(y_do=cur_batch_y_do[i],
                 y_od=cur_batch_y_od[i]) for i in range(N)
        ])
        sequences_y.append(batch_y)
    if return_y:
        return SimpleBatch(sequences), SimpleBatch(sequences_y), y_od, y_do
    else:
        return SimpleBatch(sequences), SimpleBatch(sequences_y)


def collate_wrapper_multi_branches(x_numpy, y_numpy, edge_index_list, device):
    sequences_multi_branches = []
    for edge_index in edge_index_list:
        sequences, y = collate_wrapper(x_numpy, y_numpy, edge_index, device, return_y=True)
        sequences_multi_branches.append(sequences)

    return sequences_multi_branches, y
