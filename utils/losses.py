import torch.nn.functional as F
import torch
import torch.nn as nn


class overparametrization_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, ratio_consistency=0, ratio_balance=0):
        super(overparametrization_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param()

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, outputs, label):
        # label = label.long()
        # label = torch.zeros(len(label), self.config['num_classes']).cuda().scatter_(1, label.view(-1,1), 1)
        if len(outputs) > len(index):
            output, output2 = torch.chunk(outputs, 2)
            ensembled_output = 0.5 * (output + output2).detach()
        else:
            output = outputs
            ensembled_output = output.detach()

        eps = 1e-4
        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)
        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        E = U_square - V_square
        self.E = E
        # original_prediction = F.softmax(output, dim=1)
        original_prediction = F.sigmoid(output)  # [B, C]
        prediction = torch.clamp(original_prediction + U_square - V_square.detach(), min=eps)
        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)
        label_one_hot = self.soft_to_hard(output.detach())
        MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label, reduction='sum') / len(label)  # MSE loss, to update v
        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))  # CE loss, to update u
        loss += MSE_loss  # TODO: why use BCE + MSE loss?
        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)
            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)
            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
            loss += self.ratio_balance * balance_kl

        if (len(outputs) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss(index, output, output2)
            loss += self.ratio_consistency * torch.mean(consistency_loss)

        return loss

    def consistency_loss(self, index, output1,
                         output2):  # TODO: change to multi-label. What's the purpose of consistency loss?
        preds1 = F.softmax(output1, dim=1).detach()
        preds2 = F.log_softmax(output2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv

    def soft_to_hard(self, x, threshold=0.5):
        with torch.no_grad():
            # return (torch.zeros(len(x), self.config['num_classes'])).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1),
            #                                                                  1)
            return (x > threshold).float()
