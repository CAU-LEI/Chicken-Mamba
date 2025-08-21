import torch
import torch.nn as nn
from torch.autograd import Variable

class PAM_Module(nn.Module):
    """ Position Attention Module (PAM) """
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Forward propagation.
        :param x: Input feature map (B, C, H, W)
        :return: Weighted feature map
        """
        batch_size, channels, height, width = x.size()

        # B: Query (B, H*W, C//8)
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)

        # C: Key (B, C//8, H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)

        # S: Attention Map (B, H*W, H*W)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)

        # D: Value (B, C, H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Weighting (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Add residual connection
        out = self.gamma * out + x
        return out


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def init_hidden(self, batch_size, device):
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(device))

    def forward(self, input_tensor, h_cur):
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Parameter lengths are inconsistent.")

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(input_tensor.size(1)):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], h_cur=h)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        return [self.cell_list[i].init_hidden(batch_size, device) for i in range(self.num_layers)]

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvGRURegression(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, output_dim):
        super(ConvGRURegression, self).__init__()
        self.convgru = ConvGRU(input_size, input_dim, hidden_dim, kernel_size, num_layers,
                               dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=False)
        self.pam = PAM_Module(hidden_dim[-1])
        self.fc = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        layer_output_list, _ = self.convgru(x)
        last_layer_output = layer_output_list[-1]
        last_time_step = last_layer_output[:, -1, :, :, :]
        enhanced_features = self.pam(last_time_step)
        pooled_features = torch.mean(enhanced_features, dim=[2, 3])
        output = self.fc(pooled_features)
        return output
