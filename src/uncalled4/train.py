from . import Tracks
from .align import AlignPool
from .pore_model import PoreModel, PoreModelParams
from _uncalled4 import EventDetector

import sys
from time import time
import numpy as np
import pandas as pd
from collections import Counter
import multiprocessing as mp


def init_model(tracks, k):
    p = tracks.conf.event_detector
    p.min_mean = 0
    p.max_mean = 1000000
    evdt = EventDetector(p)

    currents = list()
    length = 0

    for sam in tracks.bam_in.iter_sam():

        mv = np.array(sam.get_tag("mv"))
        mv_stride = mv[0]
        st = sam.get_tag("ts")
        moves = mv[1:]

        en = st + (np.sum(moves) * mv_stride)

        read = tracks.read_index[sam.query_name]

        c = evdt.get_means(read.signal.to_numpy()[st:en])

        currents.append(c)
        length += len(c)
        if length >= tracks.conf.train.init_events:
            break
    currents = np.concatenate(currents)

    mn = np.mean(currents)
    sd = np.std(currents)
    coef = 3
    cmin = mn-sd*coef
    cmax = mn+sd*coef

    currents = currents[(currents >= cmin) & (currents <= cmax)]

    mn = np.mean(currents)
    sd = np.std(currents)
    coef = 3
    cmin = mn-sd*coef
    cmax = mn+sd*coef

    tracks.conf.pore_model.pa_mean = np.mean(currents)
    tracks.conf.pore_model.pa_stdv = np.std(currents)
    model = PoreModel(params=tracks.conf.pore_model)
    tracks.set_model(model)

def train(conf):
    mp.set_start_method("spawn")
    #conf.tracks.load_signal = True
    conf.tracks.layers.append("moves")
    conf.mvcmp = True

    prms = conf.train

    model = None

    if len(prms.init_model) > 0:
        p = conf.pore_model
        p.name = prms.init_model
        model = PoreModel(params=p)
        if model.K != prms.kmer_len:
            dk = prms.kmer_len - model.k
            ds = p.shift - model.shift 
            if dk < 0 or ds < 0:
                raise RuntimeError("Cannot reduce k-mer length or k-mer shift")
            model = model.expand(ds,dk-ds)

    elif not prms.append and prms.kmer_len is not None:
        conf.pore_model.k = prms.kmer_len
        orig_norm_iters = conf.dtw.norm_iterations
        conf.dtw.norm_iterations = 0

    elif not prms.append:
        raise ValueError(f"Must define kmer_length, init_model, or run in append mode")
    
    if model is None:
        model=PoreModel(params=conf.pore_model)

    tracks = Tracks(model=model, conf=conf)

    _ = tracks.read_index.default_model

    if conf.dtw.norm_iterations == 0:
        sys.stderr.write("Initializing model parameters\n")
        init_model(tracks, prms.kmer_len)

    trainer = tracks.output

    if prms.append:
        conf.pore_model = trainer.model.PRMS
        tracks.set_model(trainer.model)
    else:
        trainer.set_model(tracks.model)

    if prms.skip_dtw:
        model_file = trainer.next_model(True)
        return

    bam_in = tracks.bam_in.input
    bam_in.reset()

    sys.stderr.write(f"K-mer length: {tracks.index.model.K}\n")
    
    t = time()
    for i in range(prms.iterations):
        sys.stderr.write(f"Iteration {i+1} of {prms.iterations}\n")
        bam_start = bam_in.tell()

        status_counts = Counter()
        pool = AlignPool(tracks)
        for chunk,counts in pool: #dtw_pool_iter(tracks):
            status_counts.update(counts)
            trainer.write_buffer(chunk)

            if trainer.is_full():
                break


        pool.close()

        #If EOF reached, reset BAM and read until full or entire file was read
        if not trainer.is_full():
            bam_in.reset()

            pool = AlignPool(tracks)
            for chunk,counts in pool: #dtw_pool_iter(tracks):
                status_counts.update(counts)
                trainer.write_buffer(chunk)
                if bam_in.tell() >= bam_start or trainer.is_full():
                    break
            pool.close()

        sys.stderr.write("Alignment done. Computing model...\n")
        prms.append = False
        model = trainer.next_model()
        if tracks.conf.dtw.norm_iterations == 0:
            tracks.conf.dtw.norm_iterations = orig_norm_iters

        tracks.set_model(model)

        sys.stderr.write(str(status_counts)+"\n")

    tracks.close()

