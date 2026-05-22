 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone, BackboneCLIP
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d

from .ml_decoder import MLDecoder

 
class CTranModel(nn.Module):
    def __init__(self,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False, state_weight=None, label_weight=None):
        super(CTranModel, self).__init__()
        # 是否启用lmt
        self.use_lmt = use_lmt
        # 做消融实验用的，看看没有图像信号的效果
        self.no_x_features = no_x_features # (for no image features)

        # ResNet backbone
        self.backbone = Backbone()
        # self.backbone_c = BackboneCLIP()
        self.proj1x1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        # 隐藏层，为了加上位置编码之类的
        hidden = 512 # this should match the backbone output feature size

        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))

        # Label Embeddings
        # 构建索引
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()

        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)
        self.clip_label_lt = nn.Embedding.from_pretrained(label_weight, freeze=True, padding_idx=None)
        # State Embeddings
        self.known_label_lt = nn.Embedding.from_pretrained(state_weight, freeze=True, padding_idx=0)
        # self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        # 这需要改
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            # 这也需要改
            self.position_encoding = positionalencoding2d(hidden, 7, 7).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label

        self.output_linear = torch.nn.Linear(hidden,num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        # below is just for c_tran original
        # self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

        # only use backbone
        self.is_only_backbone = False
        self.use_ml_head = False
        self.decoder = MLDecoder(num_classes=num_labels, decoder_embedding=512, initial_num_features=512)


    def forward(self, images, mask, label_emb_type='ctran', clip_emb=None, clip_model=None, return_label_emb: bool = False):

        # decide the label embedding is learnable or not
        # (B,L,hidden)
        # 标签 token（label embeddings）怎么初始化/是否可学习
        if label_emb_type == 'ctran':
            const_label_input = self.label_input.repeat(images.size(0),1).cuda()
            label_init_emb = self.label_lt(const_label_input)
        elif label_emb_type == 'onehot':
            const_label_input = F.one_hot(torch.arange(0, 17)) # (0~num_labels)
            label_init_emb = F.pad(const_label_input, pad=(0, 512 - const_label_input.shape[0], 0, 0)).unsqueeze(0)
            label_init_emb = torch.Tensor(label_init_emb).long().cuda()
        elif label_emb_type == 'clip':
            const_label_input = self.label_input.repeat(images.size(0),1).cuda()
            label_init_emb = self.clip_label_lt(const_label_input)

        features = self.backbone(images)
        features = self.proj1x1(features)  # (B, 512,  H, W)
        # (B, 512, 7, 7)，下采样没改变
        if self.downsample:
            features = self.conv_downsample(features)
        # 位置编码：(B, 512, 7, 7)
        if self.use_pos_enc:
            # 这得改
            # 嵌入位置编码
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),7,7, dtype=torch.bool).cuda())
            features = features + pos_encoding
        # 先变成(B, 512, 49)，然后变成(B, 49, 512)
        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1)

        # Convert mask values to positive integers for nn.Embedding
        # 对掩码进行处理
        label_feat_vec = custom_replace(mask,0,1,2).long()

        # Get state embeddings
        # 状态标签嵌入
        # (B,L,hidden)，状态标签也长这样
        state_embeddings = self.known_label_lt(label_feat_vec)
        init_label_embeddings = label_init_emb + state_embeddings

        if self.no_x_features:
            # (B,L,hidden)
            embeddings = init_label_embeddings
        else:
            # embeddings = (B, S+L, C)
            # embeddings = (B, 69, 512)
            embeddings = torch.cat((features, init_label_embeddings),1)
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        # 这是核心代码，送进注意力
        # embeddings：最后一层的序列特征 (B, T, C)；
        # attns：按实现要么是逐层相加后的注意力 (B, heads, T, T)。
        if not self.is_only_backbone:
            for layer in self.self_attn_layers:
                embeddings,attn = layer(embeddings,mask=None)
                attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        # (1, 17, 512)
        # 标签tokens
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        # 视觉tokens(B, 49, 512)
        # tmp_emb = embeddings[:,init_label_embeddings.size(1):,:]
        tmp_emb = embeddings[:, :embeddings.size(1) - init_label_embeddings.size(1):, :]
        # Different decoder input?
        ## (1) resnet + label embedding out
        ## (2) only label embedding perform self-attn => not better than (1)
        ## (3) embedding out directly from encoder (visual + label emb) => best now
        if self.use_ml_head:
            for i in range(label_embeddings.shape[0]):
                if i == 0:
                    output = self.decoder(tmp_emb[i].unsqueeze(0), label_embeddings[i].unsqueeze(0))
                else:
                    # output 的形状是 (B, L) → 每张图片（B）预测 L 个数值（每个标签一个概率 / logit）。
                    output = torch.cat((output, self.decoder(tmp_emb[i].unsqueeze(0), label_embeddings[i].unsqueeze(0))))
        else:
            # (1, 17, 17)
            output = self.output_linear(label_embeddings)
            diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()
            output = (output*diag_mask).sum(-1)
        if return_label_emb:
            return output, None, attns, label_embeddings
        else:
            return output, None, attns

