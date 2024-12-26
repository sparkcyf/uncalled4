from ..pore_model import PoreModel
from ..tracks import LAYER_META
from . import TrackIO
import _uncalled4
from _uncalled4 import PoreModelParams

import sys
import os
from glob import glob
import numpy as np
import pandas as pd
from time import time


#LAYERS = [("dtw",l) for l in ("kmer", "current", "stdv", "dwell")]
LAYERS = [("seq","kmer"), ("dtw","current"), ("dtw","current_sd"), ("dtw","dwell")]

class ModelTrainer(TrackIO):
    FORMAT = "model"

    def __init__(self, filename, write, tracks, track_count):
        TrackIO.__init__(self, filename, write, tracks, track_count)

        self.tprms = self.conf.train

        self.row_dtype = [
            (name, np.dtype(dt.lower() if isinstance(dt,str) else dt))
            for (_,name),dt in LAYER_META.loc[LAYERS, "dtype"].items()
        ]
        self.itemsize = sum((d.itemsize for _,d in self.row_dtype))

        self.output = self.input = None

        if self.write_mode:
            self.init_write_mode()
        else:
            self.init_read_mode()

    def init_read_mode(self, load_index=True):
        self.input = open(self._filename("data"), "rb")

        if load_index:
            self.kmer_index = pd.read_csv(self._filename("index"), index_col="kmer", sep="\t")

    def _filename(self, name, itr=None):
        if itr is None:
            itr = self.iter
        return os.path.join(self.filename, f"it{self.iter}.{name}")

    def init_write_mode(self):
        TrackIO.init_write_mode(self)

        self.model = None

        if self.tprms.append and not self.prms.buffered:
            prev_models = glob(f"{self.filename}/it*.model.npz")
            if len(prev_models) == 0:
                raise ValueError("--append can only be used with existing model training directory")

            max_itr = -1
            fname = None
            for m in prev_models:
                itr = int(os.path.basename(m).split(".")[0][2:])
                if itr > max_itr:
                    fname = m
                    max_itr = itr

            self.iter = max_itr + int(not self.tprms.skip_dtw)
            self.conf.pore_model.name = fname
            self.set_model(PoreModel(params=self.conf.pore_model))

        elif self.tprms.skip_dtw:
            self.iter = self.tprms.iterations
            self.kmer_counts = None

        else:
            os.makedirs(self.filename, exist_ok=True)
            self.conf.to_toml(os.path.join(self.filename, "conf.toml"))
            self.iter = 1
            self.kmer_counts = None

        self.buff_len = 0

        self.out_buffer = list()

    def set_model(self, model):
        self.model = model
        self.kmer_counts = pd.Series(0, index=self.model.KMERS)
        self.kmer_index = None #pd.DataFrame({"start" : 0}, index=self.model.KMERS)
        self.full_kmers = set()


    def write_alignment(self, aln):
        df = aln.to_pandas(LAYERS+["mvcmp"]).dropna()
        mask = df["mvcmp","dist"] <= self.conf.train.max_moves_dist
        dtw = df[mask][LAYERS].droplevel(0,axis=1).set_index("kmer") 

        if self.prms.buffered:
            self.out_buffer.append(dtw)
        else:
            self.write_buffer([dtw])

    def write_buffer(self, out=[], force=False):
        if len(out) > 0:
            df = pd.concat(out) \
                   .sort_index() \
                   .drop(self.full_kmers, errors="ignore") 

            kc = df.index.value_counts()
            self.kmer_counts[kc.index] += kc.to_numpy()

            full = self.kmer_counts >= self.tprms.kmer_samples
            if np.any(full):
                self.full_kmers.update(self.kmer_counts.index[full])

            self.out_buffer.append(df.reset_index().to_records(index=False,column_dtypes=dict(self.row_dtype)))
            self.buff_len += len(df)


        if self.buff_len == 0 or (self.buff_len * self.itemsize < self.tprms.buffer_size*10**6 and not force):
            return

        if self.output is None:
            self.output = open(self._filename("data"), "wb")

        out = np.concatenate(self.out_buffer)
        out.sort(kind="mergesort")

        kmers, counts = np.unique(out["kmer"], return_counts=True)
        df = pd.DataFrame({"start" : 0, "length" : counts}, index=kmers)
        df["start"].iloc[1:] = counts.cumsum()[:-1]
        if self.kmer_index is None:
            self.kmer_index = df
            df.to_csv(self._filename("index"), sep="\t", index_label="kmer", mode="w")
        else:
            df["start"] += self.kmer_index.iloc[-1].sum()
            df.to_csv(self._filename("index"), sep="\t", index_label="kmer", header=False, mode="a")
            self.kmer_index = pd.concat([self.kmer_index, df])

        self.output.write(out.tobytes())

        self.out_buffer = list()
        self.buff_len = 0

    def is_full(self):
        return self.model is not None and len(self.full_kmers) == self.model.KMER_COUNT


    def next_model(self, load_index=False):
        self.close()
        self.init_read_mode(load_index=load_index)
        self.kmer_index.sort_index(inplace=True)
        
        t = time()
        # 预分配大数组存储所有数据
        total_length = self.kmer_index['length'].sum()
        all_rows = np.zeros(total_length, dtype=self.row_dtype)
        
        # 一次性读取所有数据
        offset = 0
        starts = self.kmer_index['start'].values
        lengths = self.kmer_index['length'].values
        for start, length in zip(starts, lengths):
            self.input.seek(start * self.itemsize)
            all_rows[offset:offset + length] = np.fromfile(self.input, self.row_dtype, length)
            offset += length
        
        # 使用pandas的groupby进行快速统计
        df = pd.DataFrame(all_rows)
        if self.tprms.train_mean:
            agg_func = 'mean'
        else:
            agg_func = 'median'

        print("read done!")
            
        # 一次性计算所有统计量
        stats = df.groupby('kmer').agg({
            'current': [agg_func, 'std'],
            'current_sd': [agg_func, 'std'],
            'dwell': [agg_func, 'std'],
        })

        print("group done!")
        
        # 重组数据为所需格式，保持kmer作为列
        model_df = pd.DataFrame()
        model_df['kmer'] = stats.index
        model_df.set_index('kmer', inplace=True)
        model_df['current.mean'] = stats['current'][agg_func]
        model_df['current_sd.mean'] = stats['current_sd'][agg_func]
        model_df['dwell.mean'] = stats['dwell'][agg_func]
        model_df['current.stdv'] = stats['current']['std']
        model_df['current_sd.stdv'] = stats['current_sd']['std']
        model_df['dwell.stdv'] = stats['dwell']['std']
        model_df['count'] = df.groupby('kmer').size()
        
        # 确保包含所有KMERS
        model_df = model_df.reindex(self.model.KMERS)
        
        # 处理missing values
        if self.conf.train.init_mode == "moves_avg" and self.conf.dtw.norm_iterations == 0:
            bases = self.conf.train.moves_avg
            if bases is None:
                bases = [self.model.shift]

            for b in bases:
                model_df[b] = self.model.kmer_base(self.model.KMERS, b)
                
            base_groups = model_df.groupby(bases)
            model_df['current.mean'] = base_groups['current.mean'].transform('median')
            model_df['current.stdv'] = base_groups['current.stdv'].transform('std')
            model_df['count'] = base_groups['count'].transform('sum')

        # 处理缺失的kmer
        subs_locs = np.array([0, self.model.K-1])
        missing_kmers = model_df.index[model_df['current.mean'].isna()]
        
        if len(missing_kmers) > 0:
            for kmer in missing_kmers:
                subs = []
                for i in subs_locs:
                    old = self.model.kmer_base(kmer, i)
                    subs.extend(self.model.set_kmer_base(kmer, i, [b for b in range(4) if b != old]))
                subs = np.unique(subs)
                means = model_df.loc[subs, 'current.mean'].dropna().sort_values()
                
                if len(means) > 0:
                    mid = means.index[len(means)//2]
                    model_df.loc[kmer] = model_df.loc[mid]
                else:
                    model_df.loc[kmer] = self.model.to_df().loc[kmer]
                model_df.loc[kmer, 'count'] = 0
        
        # 保存模型
        outfile = self._filename("model.npz")
        self.model.PRMS.name = outfile
        prms = PoreModelParams(self.model.PRMS)
        
        print("save done!")
        
        # 重置索引，确保kmer作为列传入PoreModel
        model_df = model_df.reset_index()
        model_out = PoreModel(model=model_df, k=prms.k, extra_cols=True)
        model_out.PRMS = self.model.PRMS
        model_out.to_npz(outfile)
        
        self.set_model(PoreModel(params=self.model.PRMS, extra_cols=True))
        self.iter += 1
        
        return self.model


    def close(self):
        if not self.prms.buffered:
            self.write_buffer(force=True)

        if self.output is not None:
            self.output.close()
            self.output = None

        if self.input is not None:
            self.input.close()
            self.input = None

