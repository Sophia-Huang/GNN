#!/usr/bin/env python
# encoding: utf-8
"""
@Author: yulin
@Contact: huangyulinwork@163.com
@File: task1.py
@Time: 2021/6/15 22:43
@Desc:
"""

import torch
from torch_geometric.data import Data


class Network(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.authors = self.x[torch.where(self.y == 0)]
        self.papers = self.x[torch.where(self.y == 1)]
        self.institutions = self.x[torch.where(self.y == 2)]
        self.author_paper = self.edge_index.index_select(1, torch.where(self.edge_attr == 10)[0])
        self.author_institutions = self.edge_index.index_select(1, torch.where(self.edge_attr == 11)[0])

    @property
    def author_num(self):
        return len(self.authors)

    @property
    def paper_num(self):
        return len(self.papers)

    @property
    def institution_num(self):
        return len(self.institutions)

    @property
    def author_paper_relation_num(self):
        return self.author_paper.shape[1]

    @property
    def author_institutions_relation_num(self):
        return self.author_institutions.shape[1]

if __name__ == "__main__":
    """
    Author1: 0[1, 2, 3], Author2: 1[4, 5, 6]
    Paper1: 2[11, 22, 33]; Paper2: 3[44, 55, 66]
    Institution: 4[111, 222, 333] 
    
    Author-Paper: 10(0, 2), (1, 3)
    Author-Institution: 11(0, 4), (1, 4)
    """
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [11, 22, 33], [44, 55, 66], [111, 222, 333]], dtype=torch.float)
    y = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int8)
    edge_index = torch.tensor([[0, 1, 0, 1],
                               [2, 3, 4, 5]], dtype=torch.long)
    edge_attr = torch.tensor([10, 10, 11, 11],dtype=torch.int8)
    g = Network(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    # overall information
    print(f'node num: {g.num_nodes}')
    print(f'edge num: {g.num_edges}')
    print(f'contain isolated nodes: {g.contains_isolated_nodes()}')
    print(f'is directed: {g.is_directed()}')

    # specific node and relationship information
    print(f'author num: {g.author_num}')
    print(f'paper num: {g.paper_num}')
    print(f'institution num: {g.institution_num}')
    print(f'author-paper relation num: {g.author_paper_relation_num}')
    print(f'author-institution relation num: {g.author_institutions_relation_num}')

"""
# ----- Results -----
node num: 5
edge num: 4
contain isolated nodes: False
is directed: True
author num: 2
paper num: 2
institution num: 1
author-paper relation num: 2
author-institution relation num: 2

"""