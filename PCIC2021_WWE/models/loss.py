import torch
import torch.nn as nn


class IPSLoss(nn.Module):
    # for IPS-MF
    def __init__(self):
        super(IPSLoss, self).__init__()
        self.loss = 0.

    def forward(self, output, label, inverse_propensity, item, mode='nb'):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy()
        item_list = item.cpu().numpy()

        if mode == 'nb':
            weight = torch.Tensor(
                list(map(lambda x: (inverse_propensity[item_list[x]][int(label0[x])]),
                         range(0, len(label0))))).cuda()
        else:
            weight = torch.Tensor(
                list(map(lambda x: (inverse_propensity[item_list[x]]),
                         range(0, len(label0))))).cuda()

        weightedloss = torch.pow(output - label, 2) * weight
        self.loss = torch.sum(weightedloss)

        return self.loss


class SNIPSLoss(nn.Module):
    # for SNIPS-MF
    def __init__(self):
        super(SNIPSLoss, self).__init__()
        self.loss = 0.

    def forward(self, output, label, inverse_propensity, item, mode='nb'):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy()
        item_list = item.cpu().numpy()

        if mode == 'nb':
            weight = torch.Tensor(
                list(map(lambda x: (inverse_propensity[item_list[x]][int(label0[x])]),
                         range(0, len(label0))))).cuda()
        else:
            weight = torch.Tensor(
                list(map(lambda x: (inverse_propensity[item_list[x]]),
                         range(0, len(label0))))).cuda()

        weightedloss = torch.pow(output - label, 2) * weight
        normalized = torch.sum(weight)
        self.loss = torch.sum(weightedloss) / normalized

        return self.loss


class TMFLoss(nn.Module):
    # for TMF
    def __init__(self):
        super(TMFLoss, self).__init__()
        self.loss = 0.
        self.MSELoss = nn.MSELoss()

    def forward(self, tag_output, tag_label, rating_output, rating_label):
        self.loss = torch.tensor(0.0)
        loss = self.MSELoss(tag_output, tag_label) + self.MSELoss(rating_output, rating_label)
        self.loss = loss

        return self.loss


class TMFWLoss(nn.Module):
    # for TMFW
    def __init__(self):
        super(TMFWLoss, self).__init__()
        self.loss = 0.
        self.MSELoss = nn.MSELoss()

    def forward(self, tag_output, tag_label, rating_output, rating_label, weight):
        self.loss = torch.tensor(0.0)
        loss = weight * self.MSELoss(tag_output, tag_label) + self.MSELoss(rating_output, rating_label)
        self.loss = loss

        return self.loss


class TMFW1Loss(nn.Module):
    # for TMFW1
    def __init__(self):
        super(TMFW1Loss, self).__init__()
        self.loss = 0.
        self.MSELoss = nn.MSELoss()

    def forward(self, tag_output, tag_label, rating_output, rating_label, weight):
        self.loss = torch.tensor(0.0)
        loss = self.MSELoss(tag_output, tag_label) + weight * self.MSELoss(rating_output, rating_label)
        self.loss = loss

        return self.loss


class TMFPLoss(nn.Module):
    # for TMFP
    def __init__(self):
        super(TMFPLoss, self).__init__()
        self.loss = 0.
        self.MSELoss = nn.MSELoss()
        self.LogSigmoid = nn.LogSigmoid()

    def forward(self, tag_output, tag_label, rating_output, rating_label, weight1, pair_pos_output, pair_neg_output,
                weight2):
        self.loss = torch.tensor(0.0)
        loss = weight1 * self.MSELoss(
            tag_output, tag_label) + self.MSELoss(
            rating_output, rating_label) + weight2 * - self.LogSigmoid(pair_pos_output - pair_neg_output).mean()
        self.loss = loss

        return self.loss