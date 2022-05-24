from torch_geometric import nn as gnn
from torch import nn
import torch
import random
import math
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from models.OD_Net import ODNet
from models.DO_Net import DONet
from models.DualInfoTransformer import DualInfoTransformer


class Net(torch.nn.Module):

    def __init__(self, cfg, logger):
        super(Net, self).__init__()
        self.logger = logger
        self.cfg = cfg

        self.OD = ODNet(cfg, logger)
        self.DO = DONet(cfg, logger)

        self.num_nodes = cfg['model']['num_nodes']
        self.num_output_dim = cfg['model']['output_dim']
        self.num_units = cfg['model']['rnn_units']
        self.num_finished_input_dim = cfg['model']['input_dim']
        self.num_unfinished_input_dim = cfg['model']['input_dim']
        self.num_rnn_layers = cfg['model']['num_rnn_layers']

        self.seq_len = cfg['model']['seq_len']
        self.horizon = cfg['model']['horizon']
        self.head = cfg['model'].get('head', 4)
        self.d_channel = cfg['model'].get('channel', 512)

        self.use_curriculum_learning = self.cfg['model']['use_curriculum_learning']
        self.cl_decay_steps = torch.FloatTensor(data=[self.cfg['model']['cl_decay_steps']])
        self.use_input = cfg['model'].get('use_input', True)

        self.mediate_activation = nn.PReLU(self.num_units)

        self.global_step = 0

        self.encoder_first_interact = DualInfoTransformer(
                                                   h=self.head,
                                                   d_nodes=self.num_nodes,
                                                   d_model=self.num_units,
                                                   d_channel=self.d_channel)
        self.decoder_first_interact = DualInfoTransformer(
                                                   h=self.head,
                                                   d_nodes=self.num_nodes,
                                                   d_model=self.num_units,
                                                   d_channel=self.d_channel)
        self.encoder_second_interact = nn.ModuleList([DualInfoTransformer(
                                                   h=self.head,
                                                   d_nodes=self.num_nodes,
                                                   d_model=self.num_units,
                                                   d_channel=self.d_channel)
                                      for _ in range(self.num_rnn_layers - 1)])
        self.decoder_second_interact = nn.ModuleList([DualInfoTransformer(
                                                   h=self.head,
                                                   d_nodes=self.num_nodes,
                                                   d_model=self.num_units,
                                                   d_channel=self.d_channel)
                                      for _ in range(self.num_rnn_layers - 1)])


    @staticmethod
    def inverse_sigmoid_scheduler_sampling(step, k):
        """TODO: Docstring for linear_scheduler_sampling.
        :returns: TODO

        """
        try:
            return k / (k + math.exp(step / k))
        except OverflowError:
            return float('inf')

    def encoder_od_do(self,
                      sequences,
                      edge_index,
                      edge_attr=None):
        """
        Encodes input into hidden state on one branch for T steps.

        Return: hidden state on one branch."""
        enc_hiddens_od = [None] * self.num_rnn_layers
        enc_hiddens_do = [None] * self.num_rnn_layers

        finished_hidden_od = None
        long_his_hidden_od = None
        short_his_hidden_od = None

        for t, batch in enumerate(sequences):
            encoder_first_out_od, finished_hidden_od, \
            long_his_hidden_od, short_his_hidden_od, \
            enc_first_hidden_od = self.OD.encoder_first_layer(batch,
                                                              finished_hidden_od,
                                                              long_his_hidden_od,
                                                              short_his_hidden_od,
                                                              edge_index,
                                                              edge_attr)

            enc_first_out_do, enc_first_hidden_do = self.DO.encoder_first_layer(batch,
                                                                                enc_hiddens_do[0],
                                                                                edge_index,
                                                                                edge_attr)

            enc_first_interact_info_od, enc_first_interact_info_do = self.encoder_first_interact(
                                                                                enc_first_hidden_od,
                                                                                enc_first_hidden_do)

            enc_hiddens_od[0] = enc_first_hidden_od + enc_first_interact_info_od
            enc_hiddens_do[0] = enc_first_hidden_do + enc_first_interact_info_do

            enc_mid_out_od = encoder_first_out_od + enc_first_interact_info_od
            enc_mid_out_do = enc_first_out_do + enc_first_interact_info_do

            for index in range(self.num_rnn_layers - 1):
                enc_mid_out_od = self.mediate_activation(enc_mid_out_od)
                enc_mid_out_do = self.mediate_activation(enc_mid_out_do)

                enc_mid_out_od, enc_mid_hidden_od = self.OD.encoder_second_layer(index,
                                                                                 enc_mid_out_od,
                                                                                 enc_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)
                enc_mid_out_do, enc_mid_hidden_do = self.DO.encoder_second_layer(index,
                                                                                 enc_mid_out_do,
                                                                                 enc_hiddens_do[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                enc_mid_interact_info_od, enc_mid_interact_info_do = self.encoder_second_interact[index](
                                                                                 enc_mid_hidden_od,
                                                                                 enc_mid_hidden_do)

                enc_hiddens_od[index + 1] = enc_mid_hidden_od + enc_mid_interact_info_od
                enc_hiddens_do[index + 1] = enc_mid_hidden_do + enc_mid_interact_info_do

        return enc_hiddens_od, enc_hiddens_do

    def scheduled_sampling(self,
                           out,
                           label,
                           GO):
        if self.training and self.use_curriculum_learning:
            c = random.uniform(0, 1)
            T = self.inverse_sigmoid_scheduler_sampling(
                self.global_step,
                self.cl_decay_steps)
            use_truth_sequence = True if c < T else False
        else:
            use_truth_sequence = False

        if use_truth_sequence:
            # Feed the prev label as the next input
            decoder_input = label
        else:
            # detach from history as input
            decoder_input = out.detach().view(-1, self.num_output_dim)
        if not self.use_input:
            decoder_input = GO.detach()

        return decoder_input

    def decoder_od_do(self,
                      sequences,
                      enc_hiddens_od,
                      enc_hiddens_do,
                      edge_index,
                      edge_attr=None):
        predictions_od = []
        predictions_do = []

        GO_od = torch.zeros(enc_hiddens_od[0].size()[0],
                            self.num_output_dim,
                            dtype=enc_hiddens_od[0].dtype,
                            device=enc_hiddens_od[0].device)
        GO_do = torch.zeros(enc_hiddens_do[0].size()[0],
                            self.num_output_dim,
                            dtype=enc_hiddens_do[0].dtype,
                            device=enc_hiddens_do[0].device)

        dec_input_od = GO_od
        dec_hiddens_od = enc_hiddens_od

        dec_input_do = GO_do
        dec_hiddens_do = enc_hiddens_do

        for t in range(self.horizon):
            dec_first_out_od, dec_first_hidden_od = self.OD.decoder_first_layer(dec_input_od,
                                                                                dec_hiddens_od[0],
                                                                                edge_index,
                                                                                edge_attr)

            dec_first_out_do, dec_first_hidden_do = self.DO.decoder_first_layer(dec_input_do,
                                                                                dec_hiddens_do[0],
                                                                                edge_index,
                                                                                edge_attr)

            dec_first_interact_info_od, dec_first_interact_info_do = self.decoder_first_interact(
                                                                                dec_first_hidden_od,
                                                                                dec_first_hidden_do)

            dec_hiddens_od[0] = dec_first_hidden_od + dec_first_interact_info_od
            dec_hiddens_do[0] = dec_first_hidden_do + dec_first_interact_info_do
            dec_mid_out_od = dec_first_out_od + dec_first_interact_info_od
            dec_mid_out_do = dec_first_out_do + dec_first_interact_info_do

            for index in range(self.num_rnn_layers - 1):
                dec_mid_out_od = self.mediate_activation(dec_mid_out_od)
                dec_mid_out_do = self.mediate_activation(dec_mid_out_do)
                dec_mid_out_od, dec_mid_hidden_od = self.OD.decoder_second_layer(index,
                                                                                 dec_mid_out_od,
                                                                                 dec_hiddens_od[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)
                dec_mid_out_do, dec_mid_hidden_do = self.DO.decoder_second_layer(index,
                                                                                 dec_mid_out_do,
                                                                                 dec_hiddens_do[index + 1],
                                                                                 edge_index,
                                                                                 edge_attr)

                dec_second_interact_info_od, dec_second_interact_info_do = self.decoder_second_interact[index](
                                                                                 dec_mid_hidden_od,
                                                                                 dec_mid_hidden_do)

                dec_hiddens_od[index + 1] = dec_mid_hidden_od + dec_second_interact_info_od
                dec_hiddens_do[index + 1] = dec_mid_hidden_do + dec_second_interact_info_do
                dec_mid_out_od = dec_mid_out_od + dec_second_interact_info_od
                dec_mid_out_do = dec_mid_out_do + dec_second_interact_info_do

            dec_mid_out_od = dec_mid_out_od.reshape(-1, self.num_units)
            dec_mid_out_do = dec_mid_out_do.reshape(-1, self.num_units)

            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od).view(-1, self.num_nodes, self.num_output_dim)
            dec_mid_out_do = self.DO.output_layer(dec_mid_out_do).view(-1, self.num_nodes, self.num_output_dim)

            predictions_od.append(dec_mid_out_od)
            predictions_do.append(dec_mid_out_do)

            dec_input_od = self.scheduled_sampling(dec_mid_out_od, sequences[t].y_od, GO_od)
            dec_input_do = self.scheduled_sampling(dec_mid_out_do, sequences[t].y_do, GO_do)

        if self.training:
            self.global_step += 1

        return torch.stack(predictions_od).transpose(0, 1), torch.stack(predictions_do).transpose(0, 1)

    def forward(self, sequences, sequences_y):
        edge_index = sequences[0].edge_index.detach()
        edge_attr = sequences[0].edge_attr.detach()

        enc_hiddens_od, enc_hiddens_do = self.encoder_od_do(sequences,
                                                            edge_index=edge_index,
                                                            edge_attr=edge_attr)
        predictions_od, predictions_do = self.decoder_od_do(sequences_y,
                                                            enc_hiddens_od,
                                                            enc_hiddens_do,
                                                            edge_index=edge_index,
                                                            edge_attr=edge_attr)

        return predictions_od, predictions_do


