import torch.optim as optim
from model1 import *
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch

class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        """

        :optimizer: TODO
        :milestones: TODO
        :gamma: TODO
        :last_epoch: TODO
        :min_lr: TODO

        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, out_dim, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = gwnet(
            device, 
            num_nodes, 
            dropout, 
            supports=supports, 
            gcn_bool=gcn_bool, 
            addaptadj=addaptadj, 
            aptinit=aptinit, 
            in_dim=in_dim, # input feature
            out_dim=out_dim, # output feature dim
            residual_channels=nhid, 
            dilation_channels=nhid, 
            skip_channels=nhid * 8, # TODO: skip channels ?
            end_channels=nhid * 16,
            blocks=4,
        )
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = MultiStepLR(optimizer=self.optimizer, milestones=[200, 350], gamma=0.5)
        self.loss = nn.L1Loss(reduction='mean')
        self.scaler = scaler
        self.clip = 5
        self.device =device

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = input.to(self.device)
        real_val = real_val.to(self.device)
        output = self.model(input)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), predict, real

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0)).to(self.device)
        output = self.model(input)
        output = output.transpose(1,3)
        real = real_val
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real,self.device,  0.0)
        return loss.item(), predict, real