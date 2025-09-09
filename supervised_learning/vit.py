
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Vision Transformer 기반 모델 클래스 정의 (Autoencoder 스타일, but supervised reconstruction)
class ViTModel(nn.Module):
    def __init__(self, img_size=(30, 180), patch_size=5, num_classes=11, dim=64, depth=4, heads=4, mlp_dim=128):
        super(ViTModel, self).__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # 6 x 36 = 216 patches

        # Patch Embedding
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )

        # Decoder: 각 패치를 원래 픽셀 크기로 복원 후 클래스별 logit 출력
        self.decoder = nn.Linear(dim, patch_size * patch_size * num_classes)

        # 출력 reshaping 파라미터
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_patches = num_patches

    def forward(self, x):
        # x: (batch, 1, 32, 192)
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (batch, dim, 6, 36)
        x = x.flatten(2).transpose(1, 2)  # (batch, 216, dim)

        # Positional embedding 추가
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)  # (batch, 216, dim)

        # Decoder
        x = self.decoder(x)  # (batch, 216, patch_size*patch_size*num_classes)
        x = x.view(batch_size, self.num_patches, self.patch_size, self.patch_size, self.num_classes)
        x = x.permute(0, 4, 1, 2, 3)  # (batch, num_classes, num_patches_h, patch_size_h, ...)
        x = x.contiguous().view(batch_size, self.num_classes, self.img_size[0], self.img_size[1])  # (batch, num_classes, 30, 180)

        return x
