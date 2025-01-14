import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        
        # 第一个卷积层 (kernel_size=3x3)，保持特征维度
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # 第二个卷积层 (kernel_size=1x1)，提升特征维度
        self.conv2 = nn.Conv2d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
        
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        residual = x
        
        # 通过 2D 卷积进行特征提取
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        
        # 加入残差连接
        out += residual
        
        return self.relu(out)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens)
             for _ in range(num_residual_layers)]
        )
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        # 遍历残差块
        for layer in self.layers:
            x = layer(x)
        
        return self.relu(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim  # 每个嵌入向量的维度
        self._num_embeddings = num_embeddings  # 嵌入向量的数量
        self._commitment_cost = commitment_cost  # 承诺损失系数
        self.decay = decay  # EMA衰减率
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # self._embedding.weight.data.normal_()

        # 使用EMA更新codebook
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.ema_w.data.normal_()

def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)  # 输入的平方和
            + torch.sum(self._embedding.weight**2, dim=1)  # 嵌入向量的平方和
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())  # 内积
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        quantized_original = quantized

        # # 计算损失
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算平均概率和困惑度 (perplexity)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # EMA更新嵌入矩阵 (更新codebook)
        if self.training:
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encodings, dim=0)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.decay * self.ema_w + (1 - self.decay) * dw)
            
            n = torch.sum(self.ema_cluster_size)
            cluster_size = (self.ema_cluster_size + 1e-5) / (n + 1e-5) * n
            
            # 更新嵌入向量权重
            # self._embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
            self._embedding.weight.data.mul_(self.decay).add_((1 - self.decay) * dw / cluster_size.unsqueeze(1))

        return quantized, quantized_original, perplexity, encodings


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # self._embedding.weight.data.normal_()

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)
        
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)  # 输入特征平方和 (B*J, 1)
            + torch.sum(self._embedding.weight**2, dim=1)  # codebook平方和 (num_embeddings,)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())  # 内积
        )
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        quantized_original = quantized

        # 计算损失
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算平均使用概率和困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # return loss, quantized, perplexity, encodings
        return quantized, quantized_original, perplexity, encodings

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=num_hiddens // 2, 
                               kernel_size=(3,3), 
                               stride=(2,2), 
                               padding=(1,1))
        
        self.conv2 = nn.Conv2d(in_channels=num_hiddens // 2, 
                               out_channels=num_hiddens, 
                               kernel_size=(3,3), 
                               stride=(2,2), 
                               padding=(1,1))

        self.conv3 = nn.Conv2d(in_channels=num_hiddens, 
                               out_channels=num_hiddens, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        # print(f"Encoder Input (before unsqueeze): {x.shape}")
        # x = x.unsqueeze(1)
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        # print(f"Encoder Input (after unsqueeze): {x.shape}")
        x = self.conv1(x)
        x = F.relu(x)
        # print(f"Encoder Input (before unsqueeze): {x.shape}")
        x = self.conv2(x)
        x = F.relu(x)
        # print(f"Encoder Input (before unsqueeze): {x.shape}")
        x = self.conv3(x)
        # print(f"Encoder Input (before unsqueeze): {x.shape}")
        return self.residual_stack(x)
    

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=num_hiddens, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)
        
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)
        
        # self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
        #                                       out_channels=num_hiddens // 2,
        #                                       kernel_size=(4, 4), 
        #                                       stride=(2, 2), 
        #                                       padding=(1, 1),
        #                                       output_padding=(0,1))
        
        self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                              out_channels=num_hiddens // 2,
                                              kernel_size=(3, 3), 
                                              stride=(2, 2), 
                                              padding=(1, 1),
                                              output_padding=(1,0))

        self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
                                              out_channels=3,
                                              kernel_size=(3, 3), 
                                              stride=(2, 2), 
                                              padding=(1, 1),
                                              output_padding=(1,0)
                                              )
    def forward(self, x):
        # 输入的量化特征 (B, d', j, n)
        # print(f"Decoder Input (before conv1): {x.shape}")
        # 第1层卷积：提取特征 (B, d', j, n) -> (B, H, j, n)
        x = self.conv1(x)
        # print(f"Decoder Input (before conv1): {x.shape}")
        # 残差堆叠层：特征增强
        x = self.residual_stack(x)
        # print(f"Decoder Input (before conv1): {x.shape}")
        # 第1次上采样 (B, H, j, n) -> (B, H/2, 2j, 2n)
        x = self.conv_trans1(x)
        x = F.relu(x)
        # print(f"Decoder Input (before conv1): {x.shape}")
        # 第2次上采样 (B, H/2, 2j, 2n) -> (B, 1, 4j, 4n)
        x = self.conv_trans2(x)
        # print(f"Decoder Input (before conv1): {x.shape}")
        # 去掉通道维度，输出重建结果 (B, 1, J, D) -> (B, J, D)
        x =  x.squeeze(-1).permute(0, 2, 1)
        # print(f"Decoder Input (before conv1): {x.shape}")
        return x



class MotionVQVAE(nn.Module):
    # def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
    #              num_embeddings, embedding_dim, commitment_cost, decay=0,smpl=None, frame_length=None):
    def __init__(self, smpl=None, frame_length=None, num_hiddens=256, num_residual_layers=3, num_residual_hiddens=64,
        num_embeddings=1024, embedding_dim=128, commitment_cost=0.25, decay=0):
        super(MotionVQVAE, self).__init__()

        self.smpl = smpl  # 可选：未来用于生成3D形状或姿态
        self.frame_length = frame_length  # 可选：未来用于多帧时序处理

        # 编码器，将输入3D运动数据编码为低维表示
        self.encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        
        # 卷积层，用于减少通道数，使编码结果匹配VQ嵌入维度
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        # 向量量化层，根据是否使用EMA决定使用哪种量化方法
        if decay > 0.0:
            self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                             commitment_cost, decay)
        else:
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                          commitment_cost)

        # 解码器，将量化后的特征重建为3D运动序列
        self.decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e_pre_vq = self.pre_vq_conv(z_e)
        quantized, quantized_original, _, _= self.vq_vae(z_e_pre_vq)
        x_recon = self.decoder(quantized)
        return {'x_recon': x_recon, 'quantized': quantized_original, 'z_e': z_e_pre_vq}
