import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def segment_data(data:torch.Tensor, num_seg = 12):
    end = data.size(-1)
    segment_length = ceil(end/num_seg)
    if end % segment_length != 0:
        end = num_seg * segment_length
    data = F.pad(data, (0,(end-data.size(-1))), mode='replicate')
    segments = []
    for i in range(0, end, segment_length):
        segment = data[:, :, i:i + segment_length]
        segments.append(segment)
    return torch.stack(segments, dim=1)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_segments, num_heads, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, 
                                                   batch_first=True, activation=F.gelu)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim*num_segments, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.transformer(x)  # (batch, tokens, embed_dim)
        x = torch.flatten(x,1)
        return self.fc(x)  # (batch, num_classes)

class EEG_Encoder(nn.Module):
    def __init__(self, in_channels, in_size, kernel_size, hid_dim, out_dim, emb_dim, act=nn.GELU, pool=nn.AvgPool3d, groups=4):
        super(EEG_Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, hid_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            nn.GroupNorm(groups, hid_dim),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(hid_dim, hid_dim, (1, in_channels, 1)),
            act(),
            nn.GroupNorm(groups, hid_dim),
            pool(kernel_size=(1, 1, 2), stride=(1, 1, 2)),

            nn.Conv3d(hid_dim, out_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            nn.GroupNorm(groups, out_dim),
            pool(kernel_size=(1, 1, 2), stride=(1, 1, 2)),
        )
        self.embedding = nn.Linear(out_dim*in_size//4, emb_dim)

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1) # (batch, 1, num_segments, channels, segment_length) 
        x = self.conv_block(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        x = self.conv_block2(x)  # (batch, out_dim, num_segments, 1, reduced_segment_len) 
        x = x.squeeze(3).permute(0,2,1,3).flatten(2) # (batch, num_segments, out_dim * segment_length) 
        return self.embedding(x) # (batch, num_segments, emb_dim)

class fNIRS_Encoder(nn.Module):
    def __init__(self, in_channels, in_size, kernel_size, hid_dim, out_dim, emb_dim, act=nn.GELU, groups=4):
        super(fNIRS_Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(1, hid_dim, (1, 1, kernel_size), padding=(0, 0, kernel_size//2)),
            act(),
            nn.GroupNorm(groups, hid_dim),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(hid_dim, out_dim, (1, in_channels, 1)),
            act(),
            nn.GroupNorm(groups, out_dim),
        )
        self.embedding = nn.Linear(out_dim*in_size, emb_dim)

    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1) # (batch, 1, num_segments, channels, segment_length) 
        x = self.conv_block(x)  # (batch, out_dim, num_segments, channels, segment_length) 
        x = self.conv_block2(x)  # (batch, out_dim, num_segments, 1, segment_length) 
        x = x.squeeze(3).permute(0,2,1,3).flatten(2) # (batch, num_segments, out_dim * segment_length) 
        return self.embedding(x) # (batch, num_segments, emb_dim) 

class FESTA_Net(nn.Module):
    def __init__(self, 
                 eeg_shape, 
                 fnirs_shape, 
                 num_segments=12,
                 embed_dim=128, 
                 num_heads=4, 
                 num_layers=2, 
                 num_groups = 4,
                 actv_mode = "gelu", 
                 pool_mode = "mean", 
                 k_size = [13,5],
                 hid_dim = [16,32],
                 num_classes=1):
        super(FESTA_Net, self).__init__()

        self.num_segments = num_segments
        actv = dict(elu=nn.ELU, gelu=nn.GELU, relu=nn.ReLU)[actv_mode]
        pool = dict(max=nn.MaxPool3d, mean=nn.AvgPool3d)[pool_mode]

        self.eeg_enc = EEG_Encoder(eeg_shape[0], round(eeg_shape[-1]/num_segments), k_size[0], hid_dim[0], hid_dim[1], embed_dim, actv, pool, num_groups)
        self.fnirs_enc = fNIRS_Encoder(fnirs_shape[0], ceil(fnirs_shape[-1]/num_segments), k_size[1], hid_dim[0], hid_dim[1], embed_dim, actv, num_groups)
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GroupNorm(num_groups, num_segments),
        )
        self.classifier = TransformerClassifier(embed_dim, num_segments, num_heads, num_layers, num_classes)
    
    def forward(self, eeg, fnirs):
        # temporal segmentation
        eeg = segment_data(eeg, self.num_segments) # (batch, num_segments, channels, segment_length) 
        fnirs = segment_data(fnirs, self.num_segments) # (batch, num_segments, channels, segment_length)

        # modality-specific encoders
        eeg = self.eeg_enc(eeg) # (batch, num_segments, embed_dim)
        fnirs = self.fnirs_enc(fnirs) # (batch, num_segments, embed_dim)

        # multimodal feature fusion
        fused_tokens = torch.cat([eeg, fnirs], dim=2) # (batch, total_tokens, embed_dim*2)
        fused_tokens = self.multimodal_fusion(fused_tokens) # (batch, total_tokens, embed_dim)
        
        # Transformer classifier
        return self.classifier(fused_tokens) # (batch, num_classes)
