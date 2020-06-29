import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from dgfsl.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

if torch.cuda.is_available():
  dtype = torch.cuda.FloatTensor
  dtype_l = torch.cuda.LongTensor
else:
  dtype = torch.FloatTensor
  dtype_l = torch.cuda.LongTensor


class GnnNet(nn.Module):
    def __init__(self, n_way, n_support, n_query):

        super(GnnNet, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        self.feature = ConvNet(4)

        self.feat_dim = self.feature.final_feat_dim
        self.loss_fn = nn.CrossEntropyLoss()

        self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128),
                                nn.BatchNorm1d(128, track_running_stats=False))
        self.gnn = GNN(128 + self.n_way, 96, self.n_way)

        support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1).long()
        support_label = torch.zeros(self.n_way * self.n_support, self.n_way).scatter(1, support_label, 1).view(
            self.n_way, self.n_support, self.n_way)
        support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
        self.support_label = support_label.view(1, -1, self.n_way).cuda()

    def forward(self, x, is_feature=False):
        self.n_query = x.size(1) - self.n_support
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        y_query = y_query.cuda()

        if is_feature:

            assert (x.size(1) == self.n_support + 15)
            z = self.fc(x.view(-1, *x.size()[2:]))
            z = z.view(self.n_way, -1, z.size(1))
        else:
            x = x.view(-1, *x.size()[2:])
            z = self.fc(self.feature(x).view(-1, self.feat_dim))
            z = z.view(self.n_way, -1, z.size(1))

        z_stack = [
            torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1,
                                                                                                            z.size(2))
            for i in range(self.n_query)]
        assert (z_stack[0].size(1) == self.n_way * (self.n_support + 1))

        nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in z_stack], dim=0)
        scores = self.gnn(nodes)

        scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,
                                                                                                         2).contiguous().view(
            -1, self.n_way)
        loss = self.loss_fn(scores, y_query)
        return scores, loss


class GNN(nn.Module):
  def __init__(self, input_features, nf, train_N_way):
    super(GNN, self).__init__()
    self.input_features = input_features
    self.nf = nf
    self.num_layers = 2

    for i in range(self.num_layers):
      if i == 0:
        module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_l = Gconv(self.input_features, int(nf / 2), 2)
      else:
        module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
      self.add_module('layer_w{}'.format(i), module_w)
      self.add_module('layer_l{}'.format(i), module_l)

    self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
    self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, train_N_way, 2, bn_bool=False)

  def forward(self, x):
    W_init = torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)

    for i in range(self.num_layers):
      Wi = self._modules['layer_w{}'.format(i)](x, W_init)
      x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
      x = torch.cat([x, x_new], 2)

    Wl = self.w_comp_last(x, W_init)
    out = self.layer_last([Wl, x])[1]

    return out


class Gconv(nn.Module):

  def __init__(self, nf_input, nf_output, J, bn_bool=True):
    super(Gconv, self).__init__()
    self.J = J
    self.num_inputs = J*nf_input
    self.num_outputs = nf_output
    self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    self.bn_bool = bn_bool
    if self.bn_bool:
      self.bn = nn.BatchNorm1d(self.num_outputs, track_running_stats=False)

  def forward(self, input):
    W = input[0]
    x = gmul(input)

    x_size = x.size()
    x = x.contiguous()
    x = x.view(-1, self.num_inputs)
    x = self.fc(x)

    if self.bn_bool:
      x = self.bn(x)
    x = x.view(*x_size[:-1], self.num_outputs)
    return W, x


class Wcompute(nn.Module):

  def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], num_operators=1, drop=False):
    super(Wcompute, self).__init__()
    self.num_features = nf
    self.operator = operator
    self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
    self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]), track_running_stats=False)
    self.drop = drop
    if self.drop:
      self.dropout = nn.Dropout(0.3)
    self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
    self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]), track_running_stats=False)
    self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
    self.bn_3 = nn.BatchNorm2d(nf*ratio[2], track_running_stats=False)
    self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
    self.bn_4 = nn.BatchNorm2d(nf*ratio[3], track_running_stats=False)
    self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
    self.activation = activation

  def forward(self, x, W_id):
    W1 = x.unsqueeze(2)
    W2 = torch.transpose(W1, 1, 2)
    W_new = torch.abs(W1 - W2)
    W_new = torch.transpose(W_new, 1, 3)

    W_new = self.conv2d_1(W_new)
    W_new = self.bn_1(W_new)
    W_new = F.leaky_relu(W_new)
    if self.drop:
      W_new = self.dropout(W_new)

    W_new = self.conv2d_2(W_new)
    W_new = self.bn_2(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_3(W_new)
    W_new = self.bn_3(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_4(W_new)
    W_new = self.bn_4(W_new)
    W_new = F.leaky_relu(W_new)

    W_new = self.conv2d_last(W_new)
    W_new = torch.transpose(W_new, 1, 3)

    if self.activation == 'softmax':
      W_new = W_new - W_id.expand_as(W_new) * 1e8
      W_new = torch.transpose(W_new, 2, 3)

      W_new = W_new.contiguous()
      W_new_size = W_new.size()
      W_new = W_new.view(-1, W_new.size(3))
      W_new = F.softmax(W_new, dim=1)
      W_new = W_new.view(W_new_size)

      W_new = torch.transpose(W_new, 2, 3)

    elif self.activation == 'sigmoid':
      W_new = F.sigmoid(W_new)
      W_new *= (1 - W_id)
    elif self.activation == 'none':
      W_new *= (1 - W_id)
    else:
      raise (NotImplementedError)

    if self.operator == 'laplace':
      W_new = W_id - W_new
    elif self.operator == 'J2':
      W_new = torch.cat([W_id, W_new], 3)
    else:
      raise(NotImplementedError)

    return W_new


def gmul(input):

  W, x = input
  W_size = W.size()
  N = W_size[-2]
  W = W.split(1, 3)
  W = torch.cat(W, 1).squeeze(3)
  output = torch.bmm(W, x)
  output = output.split(N, 1)
  output = torch.cat(output, 2)
  return output


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

  def __init__(self, depth):
    super(ConvNet, self).__init__()
    trunk = []
    for i in range(depth):
      indim = 3 if i == 0 else 64
      outdim = 64
      B = conv3x3(indim, outdim)
      trunk.append(B)
    self.trunk = nn.Sequential(*trunk)
    self.final_feat_dim = 1600

  def forward(self, x):
    out = self.trunk(x)
    return out


if __name__ == "__main__":

    model = GnnNet(5, 5, 15).cuda()
    a = torch.randn(5, 80, 3, 84, 84).cuda()
    score, _ = model(a)
    print(score.size())



