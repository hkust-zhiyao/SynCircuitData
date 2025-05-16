import math
import torch
import torch_geometric as pyg
from datasets.data_utils import collate_fn
def loglik_nats(model, x):
    
    return - model.log_prob(x).mean()


def loglik_bpd(model, x):
    
    return -model.log_prob(x).sum() / (math.log(2) * x.num_entries)
    


def elbo_nats(model, x):
    
    return loglik_nats(model, x)


def elbo_bpd(model, x):
    
    return loglik_bpd(model, x)


def iwbo(model, x, k):
    ll = -model.nll(x)
    
    
    return torch.logsumexp(ll, dim=0) - math.log(k)


























def dataset_elbo_nats(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        nats = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            nats += elbo_nats(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), nats/count, end='\r')
    return nats / count


def dataset_elbo_bpd(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            bpd += elbo_bpd(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
    return bpd / count


























