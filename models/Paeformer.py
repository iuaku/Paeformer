import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import math
class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
       
        self.patch_length = configs.slice_len
        self.middle_size = configs.middle_len
        self.hidden_size = configs.hidden_len
        self.slice_stride = configs.slice_stride
        self.encoder_dropout = configs.encoder_dropout
        d_model = self.hidden_size
        d_ff =512
        activation="gelu"
        e_layers =1
        output_attention =0
        # self.preencoder = nn.Linear(self.patch_length, d_model),

    
        self.encoder = nn.Sequential(
            nn.Linear(self.patch_length, self.hidden_size), #hiddensiae=iTrans-dmodel
            Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=output_attention), self.hidden_size, configs.n_heads),
                    self.hidden_size,
                    configs.d_ff,
                    configs.dropout,
                    activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_size)
        )
        )


        self.decoder = nn.Linear(self.hidden_size, self.patch_length)
        # nn.Sequential(
        #     nn.Linear(self.hidden_size, self.middle_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.middle_size, self.patch_length),
        # )
        # self.num_patches = int(self.seq_len/self.patch_length)
        # self.num_patches_p = int(self.pred_len/self.patch_length)
        self.num_patches = math.ceil(self.seq_len / self.patch_length)
        self.num_patches_p = math.ceil(self.pred_len / self.patch_length)
        # self.num_patches = int(torch.ceil(torch.tensor(self.seq_len / self.patch_length)))
        # self.num_patches_p = int(torch.ceil(torch.tensor(self.pred_len / self.patch_length)))
        self.use_norm = configs.use_norm
        
        self.fc_predictor = nn.Linear(self.hidden_size*self.num_patches, self.hidden_size *self.num_patches_p)

        

    def forward(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            seq_last = x_enc[:,-1:,:].detach()
            x_enc = x_enc - seq_last

        _, L, N = x_enc.shape # B L N
        # seq_last = x[:,-1:,:].detach()
        # x = x - seq_last
        # x: [Batch, Input length, Channel] 
        x = x_enc.permute(0,2,1)

        for_enc = x.unfold(-1,self.patch_length,self.slice_stride)
        # print(for_enc.shape,x.shape)
        slices = [for_enc[:,:, i,:] for i in range(for_enc.shape[-2])]
        inputs = torch.cat(slices, dim=-1)
        encoded_slices = [self.encoder(patch) for patch in slices]
        # print("encoded_slices",encoded_slices)
        encoded_slice = torch.cat(encoded_slices, dim=-1)
        decoded_slices = [self.decoder(encoded_patch) for encoded_patch in encoded_slices]
        outputs = torch.cat(decoded_slices, dim=-1)

        data = x.chunk(self.num_patches, dim=-1)
        encoded_patche = [self.encoder(patch) for patch in data]
        encoded_patches = torch.cat(encoded_patche, dim=-1)
        prediction = self.fc_predictor(encoded_patches)
        prediction_patchs = prediction.chunk(self.num_patches_p, dim=-1)
        decoded_prediction_patches = [self.decoder(prediction_patch) for prediction_patch in prediction_patchs]
        dec_out = torch.cat(decoded_prediction_patches, dim=-1)[:,:,:self.pred_len].permute(0,2,1)

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        else:
            dec_out = dec_out+seq_last

        return dec_out[:,:self.pred_len,:],inputs,outputs 