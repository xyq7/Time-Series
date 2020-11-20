from utils.registry import MODEL_ZONE
import torch.nn as nn
import torch


@MODEL_ZONE.register
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(VanillaRNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.num_layers = num_layers
        # (batch_dim, seq_dim, feature_dim)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = None
        self.init_c = None

    def reset_hidden_state(self):
        pass

    def forward(self, x):
        self.hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, hn = self.rnn(x, self.hidden)
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    model = VanillaRNN(10, 128, 2, 64).cuda()
    x = torch.zeros((100, 12, 10)).cuda()
    out = model(x)
    print(out.size())
