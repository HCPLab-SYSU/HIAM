from torch_geometric import nn as gnn
from torch import nn
import torch
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from models.GGRUCell import GGRUCell

class DONet(torch.nn.Module):

    def __init__(self, cfg, logger):
        super(DONet, self).__init__()
        self.logger = logger
        self.cfg = cfg

        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_finished_input_dim = cfg['model']['input_dim']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']
        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.num_relations = cfg['model'].get('num_relations', 1)
        self.K = cfg['model'].get('K', 2)
        self.num_bases = cfg['model'].get('num_bases', 1)

        self.use_curriculum_learning = self.cfg['model'][
            'use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(
            data=[self.cfg['model']['cl_decay_steps']])
        self.dropout_type = cfg['model'].get('dropout_type', None)
        self.dropout_prob = cfg['model'].get('dropout_prob', 0.0)
        self.use_input = cfg['model'].get('use_input', True)

        self.global_fusion = cfg['model'].get('global_fusion', False)

        self.encoder_first_cells = GGRUCell(self.num_finished_input_dim,
                                            self.num_units,
                                            self.dropout_type,
                                            self.dropout_prob,
                                            self.num_relations,
                                            num_bases=self.num_bases,
                                            K=self.K,
                                            num_nodes=self.num_nodes,
                                            global_fusion=self.global_fusion)
        self.encoder_second_cells = nn.ModuleList([GGRUCell(self.num_units,
                                             self.num_units,
                                             self.dropout_type,
                                             self.dropout_prob,
                                             self.num_relations,
                                             num_bases=self.num_bases,
                                             K=self.K,
                                             num_nodes=self.num_nodes,
                                             global_fusion=self.global_fusion)
                                                   for _ in range(self.num_rnn_layers - 1)])

        self.decoder_first_cells = GGRUCell(self.num_finished_input_dim,
                                            self.num_units,
                                            self.dropout_type,
                                            self.dropout_prob,
                                            self.num_relations,
                                            num_bases=self.num_bases,
                                            K=self.K,
                                            num_nodes=self.num_nodes,
                                            global_fusion=self.global_fusion)

        self.decoder_second_cells = nn.ModuleList([GGRUCell(self.num_units,
                                             self.num_units,
                                             self.dropout_type,
                                             self.dropout_prob,
                                             self.num_relations,
                                             self.K,
                                             num_nodes=self.num_nodes,
                                             global_fusion=self.global_fusion)
                                                   for _ in range(self.num_rnn_layers - 1)])
        self.output_type = cfg['model'].get('output_type', 'fc')
        if self.output_type == 'fc':
            self.output_layer = nn.Linear(self.num_units, self.num_output_dim)


    def encoder_first_layer(self,
                            batch,
                            enc_first_hidden,
                            edge_index,
                            edge_attr=None):
        enc_first_out, enc_first_hidden = self.encoder_first_cells(inputs=batch.x_do,
                                                                   edge_index=edge_index,
                                                                   edge_attr=edge_attr,
                                                                   hidden=enc_first_hidden)
        return enc_first_out, enc_first_hidden

    def encoder_second_layer(self,
                             index,
                             encoder_first_out,
                             enc_second_hidden,
                             edge_index,
                             edge_attr):
        enc_second_out, enc_second_hidden = self.encoder_second_cells[index](inputs=encoder_first_out,
                                                                             hidden=enc_second_hidden,
                                                                             edge_index=edge_index,
                                                                             edge_attr=edge_attr)
        return enc_second_out, enc_second_hidden

    def decoder_first_layer(self,
                            decoder_input,
                            dec_first_hidden,
                            edge_index,
                            edge_attr=None):
        dec_first_out, dec_first_hidden = self.decoder_first_cells(inputs=decoder_input,
                                                                   hidden=dec_first_hidden,
                                                                   edge_index=edge_index,
                                                                   edge_attr=edge_attr)

        return dec_first_out, dec_first_hidden

    def decoder_second_layer(self,
                             index,
                             decoder_first_out,
                             dec_second_hidden,
                             edge_index,
                             edge_attr=None):
        dec_second_out, dec_second_hidden = self.decoder_second_cells[index](inputs=decoder_first_out,
                                                                             hidden=dec_second_hidden,
                                                                             edge_index=edge_index,
                                                                             edge_attr=edge_attr)
        return dec_second_out, dec_second_hidden
