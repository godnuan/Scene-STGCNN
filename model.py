import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

    def forward(self, x, A):
        '''
        Args:
            x: tensor of shape (batch, 2, obs_len, num_peds)
            A: tensor of shape (obs_len, num_peds, num_peds)
        Returns:
            x: tensor of shape (batch, 5, obs_len, num_peds)
        '''
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous()


class ST_Conv(nn.Module):

    def __init__(self, out_channels, seq_len):
        super(ST_Conv, self).__init__()

        self.tcn_in = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.scn = nn.Conv2d(seq_len, seq_len, kernel_size=3, padding=1)
        self.tcn_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch, out_channels, seq_len, num_peds)
        Returns:
            x: tensor of shape (batch, out_channels, seq_len, num_peds)
        '''
        x_g = self.tcn_in(x)
        x = x * torch.sigmoid(x_g)
        x = self.scn(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_g = self.tcn_out(x)
        x = x * torch.sigmoid(x_g)

        return x


class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 classes,
                 out_channels,
                 seq_len,
                 dropout=0
                 ):
        super(st_gcn, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, seq_len)
        self.scn = ConvTemporalGraphical(classes, out_channels, seq_len)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        self.st_ouput = ST_Conv(out_channels, seq_len)
        self.prelu = nn.PReLU()

    def forward(self, x, A, x_scene):
        '''
        Args:
            x: tensor of shape (batch, 2, obs_len, num_peds)
            A: tensor of shape (obs_len, num_peds, num_peds)
            x_scene: tensor of shape (batch, classes, obs_len, num_peds)
        Returns:
            x_out: tensor of shape (batch, 5, obs_len, num_peds)
        '''
        res = self.residual(x)

        x_out = self.gcn(x, A) * self.scn(x_scene, A)

        x_out = self.tcn(x_out) + res

        x_out = self.st_ouput(x_out)
        x_out = self.prelu(x_out)

        return x_out


class tcnns(nn.Module):
    def __init__(self, n_tcnns, seq_len, pred_seq_len):
        super(tcnns, self).__init__()

        self.n_layers = n_tcnns

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for j in range(1, self.n_layers):
            self.layers.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))

        self.tcnns_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for j in range(self.n_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, v):
        '''
        Args:
            v: tensor of shape (batch, obs_len, 5, num_peds)
        Returns:
            v: tensor of shape (batch, pred_len, 5, num_peds)
        '''

        v = self.prelus[0](self.layers[0](v))

        for k in range(1, self.n_layers - 1):
            v = self.prelus[k](self.layers[k](v)) + v

        v = self.tcnns_ouput(v)

        return v


class scene_stgcnn(nn.Module):
    def __init__(self, n_tcnns=1, input_feat=2, output_feat=5, classes=2, seq_len=8, pred_seq_len=12):
        super(scene_stgcnn, self).__init__()

        self.st_gcns = st_gcn(input_feat, classes, output_feat, seq_len)

        self.tcnns = tcnns(n_tcnns, seq_len, pred_seq_len)

    def forward(self, v, a, v_scene):
        '''
        Args:
            v: tensor of shape (batch, 2, obs_len, num_peds)
            a: tensor of shape (obs_len, num_peds, num_peds)
            v_scene: tensor of shape (batch, classes, obs_len, num_peds)
        Returns:
            v: tensor of shape (batch, 5, pred_len, num_peds)
            a: tensor of shape (obs_len, num_peds, num_peds)
        '''

        v = self.st_gcns(v, a, v_scene)

        v = v.permute(0, 2, 1, 3)

        # v: tensor of shape (batch, pred_len, 5, num_peds)
        v = self.tcnns(v)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a
