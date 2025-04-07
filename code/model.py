import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

def choose_nonlinear(name):
    if name == 'relu':
        nonlinear = nn.ReLU()
    elif name == 'sigmoid':
        nonlinear = nn.Sigmoid()
    elif name == 'identity':
        nonlinear = nn.Identity()
    elif name == 'tanh':
        nonlinear = nn.Tanh()
    elif name == 'leaky-relu':
        nonlinear = nn.LeakyReLU()
    elif name =='Prelu':
        nonlinear = nn.PReLU()
    elif name == 'gelu':
        nonlinear = nn.GELU()
    else:
        raise NotImplementedError("Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(name))
    
    return nonlinear

class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=1, kernel_size=7, stride=1, height_dim=True, inference=False, qk_scale=2, v_scale=1):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.qk_scale = qk_scale
        self.v_scale = v_scale
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        # 修改了这里来调整q, k, v的维度
        self.kqv_conv = nn.Conv1d(in_channels, self.depth * (qk_scale * 2 + v_scale), kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * (qk_scale * 2 + v_scale))
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * (qk_scale * 2 + v_scale), kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        self.attention_weights = None  # Initialize as None

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            
        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        kqv = self.kqv_bn(kqv) # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * (self.qk_scale * 2 + self.v_scale), height), [self.dh * self.qk_scale, self.dh * self.qk_scale, self.dh * self.v_scale], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * (self.qk_scale * 2 + self.v_scale), self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh * self.qk_scale, self.dh * self.qk_scale, self.dh * self.v_scale], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.attention_weights = weights.detach()  # Save weights for later access, not as parameters
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * self.v_scale * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth , self.v_scale * 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output

class Axial_Layer_cross(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=7, stride=1, height_dim=True, inference=False):
        super(Axial_Layer_cross, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads
        
        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.v_conv = nn.Conv1d(in_channels, self.depth, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm1d(self.depth)

        self.q_conv = nn.Conv1d(in_channels, self.depth // 2, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm1d(self.depth // 2)
		
        self.k_conv = nn.Conv1d(in_channels, self.depth // 2, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm1d(self.depth // 2)


        self.kq_conv = nn.Conv1d(in_channels, self.depth, kernel_size=1, bias=False)
        self.kq_bn = nn.BatchNorm1d(self.depth)
		
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size*kernel_size))

        # later access attention weights
        self.inference = inference
        self.attention_weights = None  # 初始化为空

    def forward(self, y, x): 
        y,x = self.resize(y,x)
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
            y = y.permute(0, 3, 1, 2)  # batch_size, width, depth, height
            y_ori = y 
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            y = y.permute(0, 2, 1, 3)  # batch_size, height, depth, width
            y_ori = y  

        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)
        y = y.reshape(batch_size * width, depth, height)
        # Compute q, k, v
        k = self.k_conv(x)
        k = self.k_bn(k) # apply batch normalization on k, q, v
		
        v = self.v_conv(x)
        v = self.kq_bn(v) # apply batch normalization on k, q, v
		
        q = self.q_conv(y)
        q = self.q_bn(q) # apply batch normalization on k, q, v

        kqv = torch.cat([k, q, v], dim = 1)
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height), [self.dh // 2, self.dh // 2, self.dh], dim=2)
        #q = q.reshape(batch_size * width, self.num_heads, self.dh, height)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2, self.kernel_size, self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits) # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)
        
        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.attention_weights = weights.detach()  # 保存权重供后续访问，不作为参数
            
        attn = torch.matmul(weights, v.transpose(2,3)).transpose(2,3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)
        # output = torch.sigmoid(output) * y_ori
        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)
        
        return output
    
    def resize(self, input, skip):
        """
        Args:
            input (batch_size, C1, H_in, W_in)
            skip (batch_size, C2, H_skip, W_skip)
        Returns:
            output: (batch_size, C, H_skip, W_skip), where C = C1 + C2
        """
        (H_in, W_in), (H_skip, W_skip) = input.size()[-2:], skip.size()[-2:]
        Ph, Pw = H_skip - H_in, W_skip - W_in
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left
        input = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        return input,skip


class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, nonlinear='relu', eps=EPS):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)
        self.nonlinear = choose_nonlinear(nonlinear)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out), where H_out = H // Sh
        """
        (Kh, Kw), (Sh, Sw) = self.kernel_size, self.stride
        Dh, Dw = self.dilation
        Kh, Kw = (Kh - 1) * Dh + 1, (Kw - 1) * Dw + 1

        H, W = input.size()[-2:]
        Ph, Pw = Kh - 1 - (Sh - (H - Kh) % Sh) % Sh, Kw - 1 - (Sw - (W - Kw) % Sw) % Sw
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        input = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        x = self.conv2d(input)
        x = self.norm2d(x)
        output = self.nonlinear(x)

        return output

class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, separable=False, nonlinear='relu', eps=EPS):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.deconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.norm2d = nn.BatchNorm2d(out_channels, eps=eps)
        self.nonlinear = choose_nonlinear(nonlinear)

    def forward(self, input, skip=None):
        """
        Args:
            input (batch_size, C1, H, W)
            skip (batch_size, C2, H, W)
        Returns:
            output: (batch_size, C, H_out, W_out)
        """
        (Kh, Kw), (Sh, Sw) = self.kernel_size, self.stride
        Dh, Dw = self.dilation
        Kh, Kw = (Kh - 1) * Dh + 1, (Kw - 1) * Dw + 1

        Ph, Pw = Kh - Sh, Kw - Sw
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        if skip is not None:
            input = self.concat_skip(input, skip)

        x = self.deconv2d(input)
        x = F.pad(x, (-Pw_left, -Pw_right, -Ph_top, -Ph_bottom))
        x = self.norm2d(x)
        output = self.nonlinear(x)

        return output

    def concat_skip(self, input, skip):
        """
        Args:
            input (batch_size, C1, H_in, W_in)
            skip (batch_size, C2, H_skip, W_skip)
        Returns:
            output: (batch_size, C, H_skip, W_skip), where C = C1 + C2
        """
        (H_in, W_in), (H_skip, W_skip) = input.size()[-2:], skip.size()[-2:]
        Ph, Pw = H_skip - H_in, W_skip - W_in
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left
        input = F.pad(input, (Pw_left, Pw_right, Ph_top, Ph_bottom))
        output = torch.cat([input, skip], dim=1)

        return output
    


class TFASA(nn.Module):
    def __init__(self, channels, kernel_size, inference=False):
        super(TFASA, self).__init__()
        self.TASA  = Axial_Layer(channels, kernel_size=kernel_size[1], height_dim=False, inference=inference)
        self.FASA = Axial_Layer(channels, kernel_size=kernel_size[0], height_dim=True, inference=inference)
        self.inference = inference

    def forward(self, x):
        x_l = self.TASA(x)
        x_m = self.FASA(x)
        
        # 拼接x_l, x_m, 和原始的x在第1维度（通道维度）
        result = torch.cat([x_l, x_m], dim=1)
        
        # 如果在推理模式下，你可能还想返回原始的x_l和x_m
        if self.inference:
            return result, x_l, x_m
        
        return result

class GCA(nn.Module):
    def __init__(self, channels, kernel_size, inference=False):
        super(GCA, self).__init__()
        self.axis_Fre  =  Axial_Layer_cross(channels, kernel_size=kernel_size[1], height_dim=False, inference=inference)
        self.axis_Time =  Axial_Layer_cross(channels, kernel_size=kernel_size[0], height_dim=True, inference=inference)
    def forward(self, x, y):  
        x_f = self.axis_Fre(x, y)
        x_t = self.axis_Time(x_f, y)
        return x_t

class DASA(nn.Module):
    def __init__(self, inference=False, nfft=1024, hop_length=256):
        super().__init__()
        
        # STFT output dimensions (from dataset.py)
        # Frequency dimension is always nfft//2 + 1
        self.freq_dim = nfft // 2 + 1
        
        # For standard 16kHz audio of ~5 seconds (~80000 samples)
        # Calculate time frames using torch.stft formula with center=True (default)
        # n_frames = 1 + (audio_length // hop_length)
        audio_len_samples = 80000  # ~5 seconds at 16kHz
        self.time_dim = 1 + (audio_len_samples // hop_length)
        
        print(f"Model input dimensions - Freq: {self.freq_dim}, Time: {self.time_dim}")
        
        # Define encoder structure with fixed strides
        self.encoder = nn.ModuleList([
            EncoderBlock2d(2,  16,  kernel_size=(7,5), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            EncoderBlock2d(16, 32,  kernel_size=(7,5), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            EncoderBlock2d(32, 64,  kernel_size=(5,3), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            EncoderBlock2d(64, 128, kernel_size=(5,3), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
        ])
        self.bottleneck = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1))
        
        # Define decoder structure
        self.decoder = nn.ModuleList([
            DecoderBlock2d(128, 64,  kernel_size=(5,3), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            DecoderBlock2d(128, 32,  kernel_size=(5,3), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            DecoderBlock2d(64,  16,  kernel_size=(7,5), stride=(2,2), dilation=(1,1), nonlinear='Prelu'),
            DecoderBlock2d(32,  2,   kernel_size=(7,5), stride=(2,2), dilation=(1,1), nonlinear='identity'),
        ])
        
        # Calculate feature dimensions after each encoder layer
        # Starting with input dimensions
        freq_dims = [self.freq_dim]
        time_dims = [self.time_dim]
        
        # Apply each encoder's stride
        for i in range(4):
            # Calculate size after strided convolution
            # formula: output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
            # With padding adjusted to maintain size before stride, this simplifies to:
            # output_size = ceil(input_size / stride)
            freq_dims.append((freq_dims[-1] + 1) // 2)  # stride=2, ceiling division
            time_dims.append((time_dims[-1] + 1) // 2)  # stride=2, ceiling division
        
        print(f"Feature dimensions after encoding:")
        for i, (f, t) in enumerate(zip(freq_dims[1:], time_dims[1:])):
            print(f"  Layer {i+1}: Freq={f}, Time={t}")
        
        # LSMSA works on the input dimensions
        self.axis = nn.ModuleList([
            TFASA(1, kernel_size=(freq_dims[0], time_dims[0]), inference=inference),
        ])
        
        # GCA works on the encoded dimensions (for skip connections)
        # We use the dimensions at encoder outputs 3, 2, 1 
        self.ca = nn.ModuleList([
            GCA(64, kernel_size=(freq_dims[3], time_dims[3]), inference=inference),
            GCA(32, kernel_size=(freq_dims[2], time_dims[2]), inference=inference),
            GCA(16, kernel_size=(freq_dims[1], time_dims[1]), inference=inference),
        ])
        
        self.inference = inference
    def forward(self, input):
        att_inputs = []
        skips = []
        x = input
        if self.inference:
            x,x_f,x_t = self.axis[0](x)
        else:
            x = self.axis[0](x)
        att_inputs.append(x)
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)
            skips.append(x)
        x = self.bottleneck(x)
        # 第一层解码器不使用跳跃连接
        x = self.decoder[0](x)
        # 对于其他解码器层，使用跳跃连接
        for i, decoder in enumerate(self.decoder[1:], start=1):
            skip = skips[-i-1] if i <= len(self.decoder) - 1 else None
            skip = self.ca[i-1](x, skip)
            x = decoder(x, skip)
        output = x

        (H_in, W_in), (H_out, W_out) = input.size()[-2:], output.size()[-2:]
        Ph, Pw = H_out - H_in, W_out - W_in
        Ph_top, Pw_left = Ph // 2, Pw // 2
        Ph_bottom, Pw_right = Ph - Ph_top, Pw - Pw_left

        output = F.pad(output, (-Pw_left, -Pw_right, -Ph_top, -Ph_bottom))
        if self.inference:
            return output,att_inputs,x_f,x_t 
        return output    
    
    def get_attention_maps(self):
        attention_maps = []
        for module in self.modules():
            if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                attention_maps.append(module.attention_weights)
        return attention_maps

model = DASA()
input = torch.randn(1, 1, 513, 313)
output = model(input)
print(output.size())