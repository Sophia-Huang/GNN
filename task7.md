# 图预测任务实践

## 超大规模数据集类的创建

问题：在一些应用场景中，数据集规模超级大，很难有足够的内存完全存下所有数据。因此我们需要一个能够按需加载样本到内存的数据集类。

```python
import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

```

每份数据被单独存成了pt文件。在使用时，我们可以通过get函数，使用索引idx读取指定的pt数据文件。

## 图样本封装成批

将多个小图封装成一个大图的时候，一个核心的修改点是对节点与边的序号进行增值。

### 一般情况

当对第k个图的edge_index张量做增值时，假设前面k-1个图的累积节点数量为 n，那么对第k个图的edge_index张量的增值n 。

### 图的配对

```python
class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
```

两部分的节点应该分开增值。

### 二部图：源节点与目标节点数不一致

```python
class BipartiteData(Data):
    def __init__(self, edge_index, x_s, x_t):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
```

源节点与目标节点分开存储。

## 代码实战

**定义数据集**

PCQM4M-LSC数据集。

```python
import os
import os.path as osp

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset
import shutil

RDLogger.DisableLog('rdApp.*')

class MyPCQM4MDataset(Dataset):

    def __init__(self, root):
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'
        super(MyPCQM4MDataset, self).__init__(root)

        filepath = osp.join(root, 'raw/data.csv.gz')
        data_df = pd.read_csv(filepath)
        self.smiles_list = data_df['smiles']
        self.homolumogap_list = data_df['homolumogap']

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.move(osp.join(self.root, 'pcqm4m_kddcup2021/raw/data.csv.gz'), osp.join(self.root, 'raw/data.csv.gz'))

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, homolumogap = self.smiles_list[idx], self.homolumogap_list[idx]
        graph = smiles2graph(smiles)
        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        y = torch.Tensor([homolumogap])
        num_nodes = int(graph['num_nodes'])
        data = Data(x, edge_index, edge_attr, y, num_nodes=num_nodes)
        return data

    # 获取数据集划分
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
        return split_dict

if __name__ == "__main__":
    dataset = MyPCQM4MDataset('dataset2')
    from torch_geometric.data import DataLoader
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    for batch in tqdm(dataloader):
        pass
```

**训练模型**

```python
#!/bin/sh

python main.py  --task_name GINGraphPooling\    # 为当前试验取名
                --device 0\                     
                --num_layers 5\                 # 使用GINConv层数
                --graph_pooling sum\            # 图读出方法
                --emb_dim 256\                  # 节点嵌入维度
                --drop_ratio 0.\
                --save_test\                    # 是否对测试集做预测并保留预测结果
                --batch_size 512\
                --epochs 100\
                --weight_decay 0.00001\
                --early_stop 10\                # 当有`early_stop`个epoches验证集结果没有提升，则停止训练
                --num_workers 4\
                --dataset_root dataset          # 存放数据集的根目录
```

