import numpy as np
import torch
from libs import SW, StreamSW
import tqdm
np.random.seed(1)
torch.manual_seed(1)

n=10000
d = 2
ks = [2,5,10, 20, 50, 100, 200, 500, 1000]
times=10
L=1000
device='cuda'
for k in ks:
    print('K {}'.format(k))
    a = torch.ones(n,device=device) / n
    b = torch.ones(n,device=device) / n
    MAEs=[]
    num_floats=[]
    trues = []
    SWks=[]
    for _ in range(times):
        X = torch.randn(n,d,device=device)-1
        Y = torch.randn(n,d,device=device)+2
        thetas = torch.randn(L,d,device=device)
        thetas = thetas.data/torch.sqrt(torch.sum(thetas.data**2,dim=1,keepdim=True))
        true_SW = SW(X,Y,a,b,L=L,thetas=thetas)
        SWk = SW(X[np.random.choice(n,int(3*k+2*np.log2(n/(2/3*k)))+1)],Y[np.random.choice(n,int(3*k+2*np.log2(n/(2/3*k)))+1)],a,b,L=L,thetas=thetas)
        SWks.append(SWk.detach().cpu().numpy())
        stream_sw = StreamSW(L=L,d=d,k=k,p=2, thetas=thetas.detach().cpu().numpy())
        for i in tqdm.tqdm(range(n)):
            stream_sw.update_mu(X[i].detach().cpu().numpy().reshape(1,-1))
            stream_sw.update_nu(Y[i].detach().cpu().numpy().reshape(1,-1))
        num_mu,num_nu = stream_sw.sizes()
        value= stream_sw.compute_distance()
        MAEs.append(torch.abs(value-true_SW.cpu()).detach().cpu().numpy())
        trues.append(true_SW.cpu().detach().cpu().numpy())
        num_floats.append(num_mu/L)
    np.savetxt('result_varying_ks/MAEs_k{}.csv'.format(k), np.array(MAEs).reshape(-1,), delimiter=',')
    np.savetxt('result_varying_ks/num_points_k{}.csv'.format(k), np.array(num_floats).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ks/trueSW_k{}.csv'.format(k), np.array(trues).reshape(-1, ), delimiter=',')
    np.savetxt('result_varying_ks/SW_k{}.csv'.format(k), np.array(SWks).reshape(-1, ), delimiter=',')





