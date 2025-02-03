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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.92):
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
        
        # quantized_original = quantized

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
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
            self._embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
            # self._embedding.weight.data.mul_(self.decay).add_((1 - self.decay) * dw / cluster_size.unsqueeze(1))

        return loss, quantized, perplexity, encodings


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
        
        # quantized_original = quantized

        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        
        # 计算平均使用概率和困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized, perplexity, encodings
        # return quantized, perplexity, encodings

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=num_hiddens // 2, 
                               kernel_size=(4, 4),  # 修改为覆盖时空维度
                               stride=(2, 2),  # 确保同时下采样时间和特征维度
                               padding=(1, 1))
        
        self.conv2 = nn.Conv2d(in_channels=num_hiddens // 2, 
                               out_channels=num_hiddens, 
                               kernel_size=(4, 4),  # 修改为覆盖时空维度
                               stride=(2, 2), 
                               padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=num_hiddens, 
                               out_channels=num_hiddens, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1))
        
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)

    def forward(self, x):
        # x: (B, T, D)
        B,T,D = x.shape
        x = x.view(B, T, 24, 6)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2).contiguous()  # 调整为 (B, D, T, J)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.residual_stack(x)
        return x

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
        #                                       )
        
        # self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
        #                                       out_channels=3,
        #                                       kernel_size=(4, 4), 
        #                                       stride=(2, 2), 
        #                                       padding=(1, 1))

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = nn.Conv2d(in_channels=num_hiddens, 
                                  out_channels=num_hiddens // 2,
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = nn.Conv2d(in_channels=num_hiddens // 2, 
                                  out_channels=6,
                                  kernel_size=3, 
                                  stride=1, 
                                  padding=1)

    def forward(self, x):
        # # print("de: ", x.shape)
        # x = self.conv1(x)
        # # print("de: ",x.shape)
        # x = self.residual_stack(x)
        # x = self.conv_trans1(x)
        # # print("de: ",x.shape)
        # x = F.relu(x)
        # x = self.conv_trans2(x) # (B, D, N, J)
        # # print("de: ",x.shape)
        
        x = self.conv1(x)
        x = self.residual_stack(x)

        x = self.upsample1(x)
        x = self.conv_up1(x)

        x = self.upsample2(x)
        x = self.conv_up2(x)

        x = torch.tanh(x)  # 让 pose 保持在 [-1,1]


        x = x.permute(0, 2, 3, 1).contiguous()  # 调整为 (B, N, J, D)
        # print("de: ",x.shape)
        B, N, J, D = x.shape
        x = x.view(B, N, J*D)
        x = x.unsqueeze(2)
        # print("shape: ",x.shape)
        return x


class MotionVQVAE(nn.Module):
    def __init__(self, smpl=None, frame_length=None, num_hiddens=256, num_residual_layers=3, num_residual_hiddens=64,
        num_embeddings=1024, embedding_dim=128, commitment_cost=0.1, decay=0):
        super(MotionVQVAE, self).__init__()

        # self.smpl = smpl  # 可选：未来用于生成3D形状或姿态
        # self.frame_length = frame_length  # 可选：未来用于多帧时序处理

        # 编码器，将输入3D运动数据编码为低维表示
        self.encoder = Encoder(6, num_hiddens,
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
        # 输入: (B, T, P, D)
        B, T, P, D = x.shape

        # 分离两个人的数据
        person1 = x[:, :, 0, :]  # (B, T, D)
        person2 = x[:, :, 1, :]  # (B, T, D)

        # 分别编码两个人的数据
        person1_encoded = self.encoder(person1)  # (B, num_hiddens, 1, T)
        person2_encoded = self.encoder(person2)  # (B, num_hiddens, 1, T)

        # 分别通过 pre_vq_conv 调整维度
        person1_pre_vq = self.pre_vq_conv(person1_encoded)  # (B, embedding_dim, 1, T)
        person2_pre_vq = self.pre_vq_conv(person2_encoded)  # (B, embedding_dim, 1, T)

        # 分别量化两个人的数据
        loss1, quantized1, perplexity1, _ = self.vq_vae(person1_pre_vq)  # (B, embedding_dim, 1, T)
        loss2, quantized2, perplexity2, _ = self.vq_vae(person2_pre_vq)  # (B, embedding_dim, 1, T)
        # 分别解码两个人的数据
        person1_recon = self.decoder(quantized1)  # (B, N, D)
        person2_recon = self.decoder(quantized2)  # (B, N, D)

        loss = (loss1 + loss2)/2
        # person1_recon = person1_recon.unsqueeze(2)
        # person2_recon = person2_recon.unsqueeze(2)

        # 合并两个人的重建数据
        x_recon = torch.cat([person1_recon, person2_recon], dim=2) 

        return loss, x_recon, perplexity1, perplexity2

# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# # 1D残差块，适配3D模型的运动特征学习
# class Residual(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
#         super(Residual, self).__init__()
        
#         # 第一个卷积层 (kernel_size=3)，保持特征维度
#         self.conv1 = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=num_residual_hiddens,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False  # 去掉偏置，稳定训练
#         )
        
#         # 第二个卷积层 (kernel_size=1)，提升特征维度
#         self.conv2 = nn.Conv1d(
#             in_channels=num_residual_hiddens,
#             out_channels=num_hiddens,
#             kernel_size=1,
#             stride=1,
#             bias=False
#         )
        
#         # 激活函数
#         self.relu = nn.ReLU(True)

#     def forward(self, x):
#         # 输入残差路径
#         residual = x
        
#         # 卷积 + 激活
#         out = self.conv1(x)
#         out = self.relu(out)
        
#         out = self.conv2(out)
        
#         # 加入残差连接
#         out += residual
        
#         # 最终激活输出
#         return self.relu(out)
    
# # # 测试 Residual Block
# # x = torch.randn(32, 128, 22)  # (B, D, J) 批次32，特征维度128，关节数22
# # block = Residual(in_channels=128, num_hiddens=128, num_residual_hiddens=64)
# # output = block(x)
# # print(output.shape)  # 输出: torch.Size([32, 128, 22])


# class ResidualStack(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(ResidualStack, self).__init__()

#         # 创建 num_residual_layers 个 Residual 块
#         self.layers = nn.ModuleList(
#             [Residual(in_channels, num_hiddens, num_residual_hiddens)
#              for _ in range(num_residual_layers)]
#         )
        
#         # 激活函数 (输出层激活)
#         self.relu = nn.ReLU(True)

#     def forward(self, x):
#         # 逐层通过 Residual Block
#         for layer in self.layers:
#             x = layer(x)
        
#         # 最终输出激活
#         return self.relu(x)

# # # 测试 Residual Stack
# # x = torch.randn(32, 128, 22)  # (B, D, J)
# # stack = ResidualStack(in_channels=128, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=64)
# # output = stack(x)
# # print(output.shape)  # 输出: torch.Size([32, 128, 22])


# class Residual(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
#         super(Residual, self).__init__()
        
#         # 第一个卷积层 (kernel_size=3x3)，保持特征维度
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=num_residual_hiddens,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False
#         )
        
#         # 第二个卷积层 (kernel_size=1x1)，提升特征维度
#         self.conv2 = nn.Conv2d(
#             in_channels=num_residual_hiddens,
#             out_channels=num_hiddens,
#             kernel_size=1,
#             stride=1,
#             bias=False
#         )
        
#         self.relu = nn.ReLU(inplace = False)

#     def forward(self, x):
#         residual = x
        
#         # 通过 2D 卷积进行特征提取
#         out = self.conv1(x)
#         out = self.relu(out)
        
#         out = self.conv2(out)
        
#         # 加入残差连接
#         out += residual
        
#         return self.relu(out)

# class ResidualStack(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(ResidualStack, self).__init__()
#         self.layers = nn.ModuleList(
#             [Residual(in_channels, num_hiddens, num_residual_hiddens)
#              for _ in range(num_residual_layers)]
#         )
#         self.relu = nn.ReLU(inplace = False)

#     def forward(self, x):
#         # 遍历残差块
#         for layer in self.layers:
#             x = layer(x)
        
#         return self.relu(x)

# class VectorQuantizerEMA(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99):
#         super(VectorQuantizerEMA, self).__init__()
        
#         self._embedding_dim = embedding_dim  # 每个嵌入向量的维度
#         self._num_embeddings = num_embeddings  # 嵌入向量的数量
#         self._commitment_cost = commitment_cost  # 承诺损失系数
#         self.decay = decay  # EMA衰减率
        
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
#         # self._embedding.weight.data.normal_()

#         # 使用EMA更新codebook
#         self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
#         self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
#         self.ema_w.data.normal_()

# def forward(self, inputs):
#         input_shape = inputs.shape
#         flat_input = inputs.view(-1, self._embedding_dim)
        
#         distances = (
#             torch.sum(flat_input**2, dim=1, keepdim=True)  # 输入的平方和
#             + torch.sum(self._embedding.weight**2, dim=1)  # 嵌入向量的平方和
#             - 2 * torch.matmul(flat_input, self._embedding.weight.t())  # 内积
#         )
        
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)

#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
#         # 计算损失
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
#         loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
#         quantized = inputs + (quantized - inputs).detach()
        
#         # 计算平均概率和困惑度 (perplexity)
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
#         # EMA更新嵌入矩阵 (更新codebook)
#         if self.training:
#             self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * torch.sum(encodings, dim=0)
            
#             dw = torch.matmul(encodings.t(), flat_input)
#             self.ema_w = nn.Parameter(self.decay * self.ema_w + (1 - self.decay) * dw)
            
#             n = torch.sum(self.ema_cluster_size)
#             cluster_size = (self.ema_cluster_size + 1e-5) / (n + 1e-5) * n
            
#             # 更新嵌入向量权重
#             # self._embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
#             self._embedding.weight.data.mul_(self.decay).add_((1 - self.decay) * dw / cluster_size.unsqueeze(1))

#         return loss, quantized, perplexity, encodings


# class VectorQuantizer(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost):
#         super(VectorQuantizer, self).__init__()
        
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
#         self._commitment_cost = commitment_cost
        
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
#         # self._embedding.weight.data.normal_()

#     def forward(self, inputs):
#         input_shape = inputs.shape
#         flat_input = inputs.view(-1, self._embedding_dim)
        
#         distances = (
#             torch.sum(flat_input**2, dim=1, keepdim=True)  # 输入特征平方和 (B*J, 1)
#             + torch.sum(self._embedding.weight**2, dim=1)  # codebook平方和 (num_embeddings,)
#             - 2 * torch.matmul(flat_input, self._embedding.weight.t())  # 内积
#         )
        
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
#         # 计算损失
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # encoder loss
#         q_latent_loss = F.mse_loss(quantized, inputs.detach())  # quantizer loss
#         loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
#         quantized = inputs + (quantized - inputs).detach()
        
#         # 计算平均使用概率和困惑度
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
#         return loss, quantized, perplexity, encodings

# class Encoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(Encoder, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=in_channels, 
#                                out_channels=num_hiddens // 2, 
#                                kernel_size=4, 
#                                stride=2, 
#                                padding=1)
        
#         self.conv2 = nn.Conv2d(in_channels=num_hiddens // 2, 
#                                out_channels=num_hiddens, 
#                                kernel_size=3, 
#                                stride=2, 
#                                padding=1)

#         self.conv3 = nn.Conv2d(in_channels=num_hiddens, 
#                                out_channels=num_hiddens, 
#                                kernel_size=3, 
#                                stride=1, 
#                                padding=1)
        
#         self.residual_stack = ResidualStack(in_channels=num_hiddens,
#                                             num_hiddens=num_hiddens,
#                                             num_residual_layers=num_residual_layers,
#                                             num_residual_hiddens=num_residual_hiddens)

#     def forward(self, x):
#         # 输入的运动序列通常为 (B, D, J)
#         # 添加维度，使其适配卷积层 (B, D, J) -> (B, 1, D, J)
#         # print(f"Encoder Input (before unsqueeze): {x.shape}")
#         x = x.unsqueeze(1)
#         # print(f"Encoder Input (after unsqueeze): {x.shape}")

#         # 卷积层1：特征提取和下采样 (B, 1, D, J) -> (B, H/2, D/2, J/2)
#         x = self.conv1(x)
#         x = F.relu(x)
#         # print(f"Encoder Input (before unsqueeze): {x.shape}")
#         # 卷积层2：进一步下采样 (B, H/2, D/2, J/2) -> (B, H, D/4, J/4)
#         x = self.conv2(x)
#         x = F.relu(x)
#         # print(f"Encoder Input (before unsqueeze): {x.shape}")
#         # 卷积层3：保持空间维度 (B, H, D/4, J/4) -> (B, H, D/4, J/4)
#         x = self.conv3(x)
#         # print(f"Encoder Input (before unsqueeze): {x.shape}")
#         # 残差块处理
#         return self.residual_stack(x)
    
# # #测试
# # encoder = Encoder(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
# # x = torch.randn(32, 512, 22)  # (B, D, J)
# # output = encoder(x)
# # print(output.shape)  # 输出: torch.Size([32, 128, 128, 6])

# class Decoder(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(Decoder, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels=in_channels, 
#                                out_channels=num_hiddens, 
#                                kernel_size=3, 
#                                stride=1, 
#                                padding=1)
        
#         self.residual_stack = ResidualStack(in_channels=num_hiddens,
#                                             num_hiddens=num_hiddens,
#                                             num_residual_layers=num_residual_layers,
#                                             num_residual_hiddens=num_residual_hiddens)
        
#         self.conv_trans1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
#                                               out_channels=num_hiddens // 2,
#                                               kernel_size=3, 
#                                               stride=2, 
#                                               padding=1,
#                                               output_padding=(0,1))
        
#         self.conv_trans2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
#                                               out_channels=1,
#                                               kernel_size=4, 
#                                               stride=2, 
#                                               padding=1
#                                               )
#     def forward(self, x):
#         # 输入的量化特征 (B, d', j, n)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         # 第1层卷积：提取特征 (B, d', j, n) -> (B, H, j, n)
#         x = self.conv1(x)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         # 残差堆叠层：特征增强
#         x = self.residual_stack(x)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         # 第1次上采样 (B, H, j, n) -> (B, H/2, 2j, 2n)
#         x = self.conv_trans1(x)
#         x = F.relu(x)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         # 第2次上采样 (B, H/2, 2j, 2n) -> (B, 1, 4j, 4n)
#         x = self.conv_trans2(x)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         # 去掉通道维度，输出重建结果 (B, 1, J, D) -> (B, J, D)
#         x = x.squeeze(1)
#         # print(f"Decoder Input (before conv1): {x.shape}")
#         return x
    
# # # 测试
# # decoder = Decoder(in_channels=512, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
# # x = torch.randn(32, 512, 6, 6)  # (B, d', j, n)
# # output = decoder(x)
# # print(output.shape)  # 输出: torch.Size([32, 22, 512]) (B, J, D)


# class MotionVQVAE(nn.Module):
#     def __init__(self, smpl=None, frame_length=None, num_hiddens=256, num_residual_layers=3, num_residual_hiddens=64,
#         num_embeddings=1024, embedding_dim=128, commitment_cost=0.25, decay=0):
#         super(MotionVQVAE, self).__init__()

#         # self.smpl = smpl  # 可选：未来用于生成3D形状或姿态
#         # self.frame_length = frame_length  # 可选：未来用于多帧时序处理

#         # 编码器，将输入3D运动数据编码为低维表示
#         self.encoder = Encoder(1, num_hiddens,
#                                 num_residual_layers, 
#                                 num_residual_hiddens)
        
#         # 卷积层，用于减少通道数，使编码结果匹配VQ嵌入维度
#         self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
#                                       out_channels=embedding_dim,
#                                       kernel_size=1, 
#                                       stride=1)

#         # 向量量化层，根据是否使用EMA决定使用哪种量化方法
#         if decay > 0.0:
#             self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
#                                              commitment_cost, decay)
#         else:
#             self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
#                                           commitment_cost)

#         # 解码器，将量化后的特征重建为3D运动序列
#         self.decoder = Decoder(embedding_dim,
#                                 num_hiddens, 
#                                 num_residual_layers, 
#                                 num_residual_hiddens)

#     def forward(self, x):
#         # print(f"Input to Encoder (before): {x.shape}")
#         # 编码器提取低维表示 (B, J, D) -> (B, num_hiddens, j, n)
#         z = self.encoder(x)
        
#         # 降维卷积匹配量化输入维度 (B, num_hiddens, j, n) -> (B, embedding_dim, j, n)
#         z = self.pre_vq_conv(z)

#         # 量化层计算损失和量化输出
#         loss, quantized, perplexity, _ = self.vq_vae(z)

#         # 解码器将量化结果还原到3D运动空间
#         x_recon = self.decoder(quantized)

#         # 返回量化损失、重建运动数据和困惑度
#         return loss, x_recon, perplexity