import torch.nn as nn
import torch
import torch.nn.functional as F

# Original implementation from: https://github.com/cdancette/rubi.bootstrap.pytorch/ , under the BSD 3-Clause License
"""
BSD 3-Clause License

Copyright (c) 2019+, Remi Cadene, Corentin Dancette
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class RUBiCriterion(nn.Module):

    def __init__(self, question_loss_weight=1.0):
        super().__init__()

        self.question_loss_weight = question_loss_weight
        self.fusion_loss = nn.CrossEntropyLoss(size_average=False)
        self.question_loss = nn.CrossEntropyLoss(size_average=False)
     
    def forward(self, net_out, batch):
        out = {}
        # logits = net_out['logits']
        logits_q = net_out['logits_q']
        logits_rubi = net_out['logits_rubi']
        class_id = batch['class_id']#.squeeze(1)
        fusion_loss = self.fusion_loss(logits_rubi, class_id)
        question_loss = self.question_loss(logits_q, class_id)
        loss = fusion_loss + self.question_loss_weight * question_loss

        out['loss'] = loss
        out['loss_mm_q'] = fusion_loss
        out['loss_q'] = question_loss
        return out
