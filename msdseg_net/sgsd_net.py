# pytorch implement of pp_liteseg

import torch
import torch.nn as nn
from torch.nn import init
import math
import torch.nn.functional as F
from stdc import STDCNet813
from benchmark import benchmark_eval,benchmark_train,benchmark_memory
from model_utils import BasicBlock, Bottleneck, segmenthead,  PagFM, Light_Bag, APPM_81632


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1

class SGSD_Net(nn.Module):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.
    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".
    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone_indices=[2, 3, 4],
                 planes=64,
                 out_indices=(0, 1, 2),
                 pretrain_backbone=None,
                 pretrained="",
                 change_num_classes=False):
        super().__init__()
        self.out_indices = out_indices
        backbone = STDCNet813(pretrain_model=pretrain_backbone)
        self.backbone = backbone
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # P Branch
        self.compression3 = nn.Sequential(
                                    nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
                                    BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        self.compression4 = nn.Sequential(
                                    nn.Conv2d(planes * 16, planes * 2, kernel_size=1, bias=False),
                                    BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 4, planes * 2, 2)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, 2)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.cm = APPM_81632(backbone_out_chs[-1], branch_planes=128, outplanes=128)
        self.dfm = Light_Bag(planes * 2, planes * 2)

        self.final_layer = segmenthead(planes * 2, planes * 2, num_classes)

        if pretrained != "":
            print('use pretrain model {}'.format(pretrained))
            dic = torch.load(pretrained, map_location='cpu')
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_hw = x.shape[2:]
        x2, x4, x8, x16, x32 = self.backbone(x)
        x_ = self.layer3_(x8)   # [2, 64, 64, 128]  8
        x_ = self.pag3(x_, self.compression3(x16))  # [2, 128, 64, 128]

        x_ = self.layer4_(self.relu(x_))  # [2, 128, 64, 128]
        x_ = self.pag4(x_, self.compression4(x32))  # [2, 128, 64, 128]

        x_ = self.layer5_(self.relu(x_))  # [2, 128, 64, 128]
        cm = F.interpolate(self.cm(x32), size=x_.shape[2:],
                            mode='bilinear', align_corners=False)  # [2, 128, 16, 32]
        out = self.dfm(x_, cm)
        out = F.interpolate(self.final_layer(out), size=x_hw,
                            mode='bilinear', align_corners=False)  # [2, 128, 16, 32]
        return out

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def calculate_flops():
    from fvcore.nn import FlopCountAnalysis, flop_count_table, ActivationCountAnalysis
    model1=SGSD_Net(19).eval()
    print(model1)
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23,get_ddrnet_23slim
    x=torch.randn(1, 3, 512, 1024)
    model2=get_ddrnet_23().eval()
    for model in [model1,model2]:
        flops = FlopCountAnalysis(model, x)
        print(flop_count_table(flops))

def calculate_params(model):
    #https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = model.parameters()
    params2 = sum([np.prod(p.size()) for p in model_parameters])
    return params, params2

def cityscapes_speed_test():
    print("cityscapes speed test")
    # from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    from competitors_models.bisenetv2 import BiSeNetV2
    regseg=SGSD_Net(19).eval()
    # print(regseg)
    # ddrnet23=get_ddrnet_23()
    bisenetv2 = BiSeNetV2(19).eval()
    x=torch.randn(1, 3, 1024, 2048)
    ts=[]
    # ts.extend(benchmark_eval([regseg, ddrnet23, bisenetv2],x,True))
    ts.extend(benchmark_eval([regseg, bisenetv2], x, True))
    print(ts)

def camvid_speed_test():
    print("camvid speed test")
    from competitors_models.DDRNet_Reimplementation import get_ddrnet_23
    from competitors_models.bisenetv2 import BiSeNetV2
    regseg=SGSD_Net(11).eval()
    # bisenetv2 = BiSeNetV2(11).eval()
    ddrnet23=get_ddrnet_23(num_classes=11).eval()
    x=torch.randn(1,3,720,960)
    ts=[]
    ts.extend(benchmark_eval([regseg, ddrnet23],x,True))
    # ts.extend(benchmark_eval([regseg, bisenetv2], x, True))
    print(ts)

if __name__ == "__main__":
    # cityscapes_speed_test()
    # camvid_speed_test()
    calculate_flops()
    # dilation_speed_test()
    # block_speed_test()
    # calculate_params(model)
