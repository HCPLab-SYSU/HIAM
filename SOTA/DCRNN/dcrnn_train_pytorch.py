from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import pickle
import numpy as np

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor

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

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        # sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        adj_mx = load_pickle(graph_pkl_filename).astype(np.float32)

        for i in range(len(adj_mx)):  # 构建邻接矩阵
            adj_mx[i, i] = 0

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
