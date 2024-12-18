#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

from facebank import load_facebank

device = torch.device("cpu")
cur_path = os.path.dirname(os.path.realpath(__file__)) + '/'

def load_facebank_data():
    targets, names = load_facebank(path=cur_path+'../facebank')
    print('Facebank Loaded')
    return targets, names

def face_recognition(embs, targets, thres, names):
        source_embs = torch.cat(embs)  # number of detected faces x 512
        diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
        dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
        minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
        min_idx[minimum > ((thres-156)/(-80))] = -1  # if no match, set idx to -1
        score = minimum
        results = min_idx
    
        names[0] = 'unknown'
        return results, score