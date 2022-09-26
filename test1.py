import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import networkx as nx
import scipy.sparse
import numpy as np

print(dgl.__version__)
sampler = dgl.dataloading.NeighborSampler([20,30])
#sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

#dataset = dgl.data.SSTDataset()
dataset = dgl.data.CiteseerGraphDataset()
#dataset = dgl.data.PubmedGraphDataset()
#dataset = dgl.data.CoraFullDataset()
#dataset = dgl.data.RedditDataset()
#dataset = dgl.data.rdf.AMDataset()

# adj = scipy.sparse.load_npz('/home/hda/padata/enwiki/adj.npz')
# g = dgl.from_scipy(adj)
# g = dgl.add_self_loop(g)
# node_features = np.load('/home/hda/padata/enwiki/feat.npy')
# node_labels = np.load('/home/hda/padata/enwiki/labels.npy')
# g.ndata['feat'] = torch.tensor(node_features)
# g.ndata['label'] = torch.tensor(node_labels)
# #g = g.to('cuda:0')
# train_nids = g.nodes()
# train_nids = train_nids.to('cuda:0')

# n_features = node_features.shape[1]
# n_labels = int(node_labels.max().item() + 1)

# print(g)
print(dataset[0])
g = dataset[0]
#g = g.to('cuda:0')
train_nids = g.nodes()
train_nids = train_nids.to('cuda:0')
node_features = g.ndata['feat']
node_labels = g.ndata['label']


n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)
#train_mask = g.ndata['train_mask']
#train_nids = train_nids[train_mask]

dataloader = dgl.dataloading.DataLoader(
    g,train_nids,sampler,
    device = torch.device('cuda:0'),
    batch_size = 2048,
    shuffle =True,
    drop_last = False,
    num_workers = 0,
    use_uva=True
)
# start=time.time()
# inport_nodes,output_nodes,blocks = next(iter(dataloader))
# end = time.time()
# print("运行时间:%.2f秒"%(end-start))
class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x
model = StochasticTwoLayerGCN(in_features=n_features, hidden_features=100, out_features=n_labels)
model = model.to('cuda:0')
opt = torch.optim.Adam(model.parameters())
start=time.time()
for epoch in range(10):
    epochstartime = time.time()
    time1 = 0
    time2 = 0
    time3 = 0
    for input_nodes, output_nodes, blocks in dataloader:
        data_trans_start = time.time()
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        data_trans_end = time.time()
        time1 +=(data_trans_end-data_trans_start)
        
        input_features = blocks[0].srcdata['feat']
        output_labels = blocks[-1].dstdata['label']
        agg_start = time.time()
        output_predictions = model(blocks, input_features)
        agg_end = time.time()
        time2+=(agg_end-agg_start)
        

        back_start = time.time()
        loss = F.cross_entropy(output_predictions,output_labels)
        back_end = time.time()
        time3 +=(back_end-back_start)
        

        opt.zero_grad()
        loss.backward()
        opt.step()
    print("loss:",loss)
    epochendtime = time.time()

    print("数据传输时间:",time1)
    print("聚合时间:",time2)
    print("反向时间:",time3)

    print("epoch时间:%.2f秒"%(epochendtime-epochstartime))
end = time.time()
print("运行时间:%.2f秒"%(end-start))