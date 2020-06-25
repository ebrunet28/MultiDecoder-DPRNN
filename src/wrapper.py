import torch
EPS = 1e-8

class Wrapper(torch.nn.Module):
    def __init__(self, base, fixer, iterations):
        super().__init__()
        self.base = base
        self.fixer = fixer
        self.iterations = iterations
        #self.alpha = torch.nn.Parameter(torch.Tensor([0.5]))
        
        
    def unit_norm(self, tensor):
        # tensor.shape = [batch_size, C, length]
        mean = tensor.mean(dim = -1, keepdim = True) # [batch_size, C, 1]
        var = torch.var(tensor, dim = -1, keepdim=True, unbiased=False)  # [batch_size, C, 1]
        tensor = (tensor - mean) / torch.pow(var + EPS, 0.5)
        return tensor
        
    def forward(self, x):
        # x: [batch_size, 1, length]
        shortcut = x
        out_list = []
        for i in range(self.iterations):
            if i > 0:
                #x = shortcut*self.alpha+self.unit_norm(x)*(1-self.alpha)
                x = torch.cat((shortcut, self.unit_norm(x)), dim = 1)
                x = self.fixer(x)
            else:
                x = self.base(x)
            out_list.append(x)
        return out_list
