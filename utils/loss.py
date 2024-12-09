
import torch


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


class CarCFDLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(CarCFDLoss, self).__init__()

        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)

    def compute_loss(self, x, y, batch, graph, sep=True, **kwargs):
        mask = (graph.batch == batch)
        surf = graph.surf[mask]
        press_loss = self.lp_loss(x[:, surf, -1], y[:, surf, -1])
        vol_loss = self.lp_loss(x[:, :, :-1], y[:, :, :-1])
        
        if sep:
            return [press_loss + vol_loss, press_loss, vol_loss]
        else:
            return press_loss + vol_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)


class MultipleLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        
        self.lp_loss = LpLoss(d=d, p=p, size_average=size_average, reduction=reduction)
    
    def compute_loss(self, x, y, sep=True, **kwargs):
        num_feature = x.size(2)
        loss_list = []
        for i in range(num_feature):
            loss_list.append(self.lp_loss(x[:, :, i], y[:, :, i]))
        
        all_loss = sum(loss_list)
        
        if sep:
            return [all_loss] + loss_list
        else:
            return all_loss
    
    def __call__(self, x, y, **kwargs):
        return self.compute_loss(x, y, **kwargs)