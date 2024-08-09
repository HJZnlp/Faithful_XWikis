# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from .noise_segment_embedding import LearnedNoiseSegmentEmbedding

def NoiseSegmentEmbedding(
        # num_embeddings: int,
        embedding_dim: int,
        # padding_idx: int,
        # learned: bool = False,
):
    
    # m = LearnedNoiseSegmentEmbedding(2, embedding_dim)
    # nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    m = F.embedding(2,embedding_dim)
    
   
    return m
