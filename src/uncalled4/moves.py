#!/usr/bin/env python3

import sys, os
import numpy as np
import argparse
from collections import defaultdict, namedtuple
import re
import time
import pandas as pd
import scipy.stats
import copy
import pysam

from _uncalled4 import _AlnDF, IntervalIndexI64, IntervalIndexI32, moves_to_aln, read_to_ref_moves
from .tracks import AlnDF
from .pore_model import PoreModel
from .config import Config
from .argparse import Opt
from .ref_index import RefCoord
from .params import BASECALLER_PROFILES

MOVE_SHIFTS = {
   "dna_r10.3_450bps" : 3,
   "dna_r10.3_450bpsm" : 3,
   "dna_r10.4.1_e8.2_260bps" : 4,
   "dna_r10.4.1_e8.2_400bps" : 4,
   "dna_r10_450bps" : 4,
   "dna_r10.4_e8.1" : 4,
   "dna_r10.4_e8.1m" : 4,
   "dna_r9.4.1_450bps" : 3,
   "dna_r9.4.1_e8.1" : 3,
   "dna_r9.4.1_e8.1m" : 3,
   "dna_r9.5_450bps" : 3,
   "rna_r9.4.1_70bps" : 2,
   "rna004_130bps" : 3
}
INT32_NA = np.iinfo(np.int32).max

def sam_to_read_moves(conf, read, sam, shift=False):
    if read.bc_loaded:
        mv_stride = read.move_stride
        moves = np.array(read.moves)
        template_start = read.template_start

    elif sam.has_tag("mv"):
        mv = np.array(sam.get_tag("mv"))
        mv_stride = mv[0]
        template_start = sam.get_tag("ts")
        moves = mv[1:]

    else:
        return None
        
    aln = moves_to_aln(moves, template_start, mv_stride)

    if shift:
        shift = BASECALLER_PROFILES.get(conf.tracks.basecaller_profile, None)
        aln.index.shift(-shift+1)
        aln = aln.slice(shift, len(aln)-shift+1)
    return aln

def sam_to_ref_moves(conf, ref_index, read, sam):
    if read is None:# or read.empty(): 
        return None
    if conf.tracks.zero_ts:
        sam.set_tag("ts",0)
    read_moves = sam_to_read_moves(conf, read, sam)
    if read_moves is None:
        return None

    shift = BASECALLER_PROFILES.get(conf.tracks.basecaller_profile, None)
    if shift is None:
        raise ValueError("Failed to detect basecaller profile\nPlease specify valid '--basecaller-profile [preset]', '--pore-model [preset]', or '--flowcell [flowcell_id] --kit [kit_id]'")

    model = ref_index.model

    is_fwd = not sam.is_reverse
    flip_ref = is_fwd == model.reverse

    cig = sam.cigartuples
    start_shift = cig[0][1] if cig[0][0] == pysam.CHARD_CLIP else 0
    end_shift = cig[-1][1] if cig[-1][0] == pysam.CHARD_CLIP else 0

    ar = np.array(sam.get_aligned_pairs())
    ins = ar[:,1] == None
    ar = ar[ar[:,1] != None] #TODO keep track of insertion counts

    if flip_ref:
        ar = ar[::-1]
        qrys = ar[:,0] 
        read_len = sam.infer_query_length()
        na = qrys == None
        qrys[na] = 0
        qrys = read_len - qrys - 1 + end_shift
        qrys[na] = INT32_NA
    else:
        qrys = ar[:,0]
        na = qrys == None
        qrys[na] = 0
        qrys[na] = INT32_NA

    refs = np.array(ref_index.pos_to_mpos(ar[:,1], is_fwd))
    qrys = np.array(qrys, dtype=np.int64)

    ref_moves = read_to_ref_moves(read_moves, refs, qrys, conf.dtw.del_max, conf.dtw.ins_max, True)
    ref_moves.index.shift(-shift)

    ret = ref_moves.slice(shift, len(ref_moves)-shift)

    return ret
