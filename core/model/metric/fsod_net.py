# -*- coding: utf-8 -*-
"""
@inproceedings{DBLP:conf/nips/SnellSZ17,
  author    = {Jake Snell and
               Kevin Swersky and
               Richard S. Zemel},
  title     = {Prototypical Networks for Few-shot Learning},
  booktitle = {Advances in Neural Information Processing Systems 30: Annual Conference
               on Neural Information Processing Systems 2017, December 4-9, 2017,
               Long Beach, CA, {USA}},
  pages     = {4077--4087},
  year      = {2017},
  url       = {https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html}
}
https://arxiv.org/abs/1703.05175

Adapted from https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch.
"""
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init

from core.utils import accuracy
from .metric_model import MetricModel


class FSODLayer(nn.Module):
    def __init__(self, dim_in=2048):
        super(FSODLayer, self).__init__()

        self.patch_relation = True
        self.local_correlation = True
        self.global_relation = True

        if self.patch_relation:
            self.conv_1 = nn.Conv2d(dim_in * 2, int(dim_in / 4), 1, padding=0, bias=False)
            self.conv_2 = nn.Conv2d(int(dim_in / 4), int(dim_in / 4), 3, padding=0, bias=False)
            self.conv_3 = nn.Conv2d(int(dim_in / 4), dim_in, 1, padding=0, bias=False)
            self.bbox_pred_pr = nn.Linear(dim_in, 4 * 2)
            self.cls_score_pr = nn.Linear(dim_in, 2)  # nn.Linear(dim_in, 2)

        if self.local_correlation:
            self.conv_cor = nn.Conv2d(dim_in, dim_in, 1, padding=0, bias=False)
            # self.bbox_pred_cor = nn.Linear(dim_in, 4 * 2)
            self.cls_score_cor = nn.Linear(dim_in, 2)  # nn.Linear(dim_in, 2)

        if self.global_relation:
            self.fc_1 = nn.Linear(dim_in * 2, dim_in)
            self.fc_2 = nn.Linear(dim_in, dim_in)
            self.cls_score_fc = nn.Linear(dim_in, 2)  # nn.Linear(dim_in, 2)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.avgpool_fc = nn.AvgPool2d(7)
        
        self._init_weights()

    def forward(
            self,
            x_query,
            x_support
    ):
        # multi-correlation

        x_query_fc = self.avgpool_fc(x_query).squeeze(3).squeeze(2)

        x_query_cor = self.conv_cor(x_query)

        support = x_support.mean(0, True)
        # fc
        support_fc = self.avgpool_fc(support).squeeze(3).squeeze(2).expand_as(x_query_fc)
        cat_fc = torch.cat((x_query_fc, support_fc), 1)
        out_fc = F.relu(self.fc_1(cat_fc), inplace=True)
        out_fc = F.relu(self.fc_2(out_fc), inplace=True)
        cls_score_fc = self.cls_score_fc(out_fc)

        # correlation

        support_cor = self.conv_cor(support)
        x_cor = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=2048), inplace=True).squeeze(3).squeeze(2)
        cls_score_cor = self.cls_score_cor(x_cor)

        # relation

        support_relation = support.expand_as(x_query)
        x = torch.cat((x_query, support_relation), 1)
        x = F.relu(self.conv_1(x), inplace=True) # 5x5
        x = self.avgpool(x)
        x = F.relu(self.conv_2(x), inplace=True) # 3x3
        x = F.relu(self.conv_3(x), inplace=True) # 3x3
        x = self.avgpool(x) # 1x1
        x = x.squeeze(3).squeeze(2)
        cls_score_pr = self.cls_score_pr(x)

        # final result
        cls_score_all = cls_score_pr + cls_score_cor + cls_score_fc
        if not self.training:
            cls_score_all = F.softmax(cls_score_all, dim=1)
        
        return cls_score_all
    
    def _init_weights(self):

        if self.patch_relation:
            init.normal_(self.conv_1.weight, std=0.01)
            init.normal_(self.conv_2.weight, std=0.01)
            init.normal_(self.conv_3.weight, std=0.01)
            init.normal_(self.cls_score_pr.weight, std=0.01)
            init.constant_(self.cls_score_pr.bias, 0)

        if self.local_correlation:
            init.normal_(self.conv_cor.weight, std=0.01)
            init.normal_(self.cls_score_cor.weight, std=0.01)
            init.constant_(self.cls_score_cor.bias, 0)

        if self.global_relation:
            init.normal_(self.fc_1.weight, std=0.01)
            init.constant_(self.fc_1.bias, 0)
            init.normal_(self.fc_2.weight, std=0.01)
            init.constant_(self.fc_2.bias, 0)
            init.normal_(self.cls_score_fc.weight, std=0.01)
            init.constant_(self.cls_score_fc.bias, 0)


class ProtoFSODLayer(nn.Module):
    def __init__(self, dim_in=2048):
        super(ProtoFSODLayer, self).__init__()

        self.ln_in = nn.Sequential(
            nn.Linear(1600, 1600),
            nn.ReLU(inplace=True), 
            nn.Linear(1600, 2048 * 7 * 7)
        )

        self.fsod_layer = FSODLayer()

    def forward(
            self,
            query_feat,
            support_feat,
            way_num,
            shot_num,
            query_num,
            mode="euclidean",
    ):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # # t, wq, c
        # query_feat = query_feat.view(t * way_num, query_num, c)
        # # t, w, c
        # support_feat = support_feat.view(t * way_num, shot_num, c)

        query_feat = self.ln_in(query_feat).view(t, way_num * query_num, 2048, 7, 7)
        support_feat = self.ln_in(support_feat).view(t, way_num, shot_num, 2048, 7, 7)

        # multi-correlation
        all_scores = []
        for i in range(support_feat.shape[0]):
            for j in range(support_feat.shape[1]):
                score = self.fsod_layer(query_feat[i], support_feat[i, j]).unsqueeze(0)

                all_scores.append(score)

        all_scores = torch.cat(all_scores).view(t, way_num, way_num * query_num, 2)[..., 1].transpose(-1, -2)

        return all_scores


class ProtoFSODNet(MetricModel):
    def __init__(self, **kwargs):
        super(ProtoFSODNet, self).__init__(**kwargs)
        self.proto_fsod_layer = ProtoFSODLayer()
        self.loss_func = nn.CrossEntropyLoss()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
                self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.proto_fsod_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
                self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            emb, mode=1
        )

        output = self.proto_fsod_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)

        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
