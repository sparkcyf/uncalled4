import os
import collections.abc

import sys
import numpy as np
import pandas as pd
import h5py
import itertools

from _uncalled4 import PoreModelParams, ArrayU32, ArrayU16
import _uncalled4

from . import config, RefCoord

CACHE = dict()

PARAM_TYPES = {
    "name" :  str,
    "k" : np.int32,
    "shift" :   np.int32,
    "pa_mean" : np.float32, 
    "pa_stdv" : np.float32, 
    "norm_max" : np.float32,
    "sample_rate" : np.float32,
    "bases_per_sec" : np.float32,
    "reverse" : bool,
    "complement" : bool,
    "flowcell" : str,
    "kit" : str
}

#https://droog.gs.washington.edu/parc/images/iupac.html
NT_CODES = {
    "A" : [0],
    "C" : [1],
    "G" : [2],
    "T" : [3],
    "M" : [0, 1],
    "R" : [0, 2],
    "W" : [0, 3],
    "S" : [1, 2],
    "Y" : [1, 3],
    "K" : [2, 3],
    "V" : [0, 1, 2],
    "H" : [0, 1, 3],
    "D" : [0, 2, 3],
    "B" : [1, 2, 3],
    "N" : None
}


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

class PoreModel:

    BASES = np.array(["A","C","G","T"])

    PRESET_DIR = os.path.join(ROOT_DIR, "models")
    PRESET_EXT = ".npz"

    PRESET_MAP = None
    PRESETS = ["dna_r10.4.1_400bps_9mer", "dna_r9.4.1_400bps_6mer", "rna_r9.4.1_70bps_5mer", "rna004_130bps_9mer", "tombo/rna_r9.4.1_70bps_5mer", "tombo/dna_r9.4.1_400bps_6mer"]
    PRESETS_STR = "'" + "', '".join(PRESETS) + "'"

    @classmethod
    def _init_presets(cls):
        if cls.PRESET_MAP is None:
            df = pd.read_csv(
                os.path.join(cls.PRESET_DIR, "presets.tsv"), 
                sep="\t", index_col=("flowcell","kit")
            ).sort_index()

            cls.PRESET_MAP = df[df["preset_model"] != "_"]

    @staticmethod
    def _param_defaults():
        return PoreModelParams(config._DEFAULTS.pore_model)

    @staticmethod
    def get_kmer_shift(k):
        return (k - 1) // 2

    #TODO load params like normal, maybe need python wrapper
    #store in pore model comments or binary
    #usually pass params, optionaly df/cache
    def __init__(self, *args, model=None, df=None, extra_cols=True, cache=True, normalize=True, **kwargs):
        self.conf, prms = config._init_group(
            "pore_model", _param_names=["name", "k", "shift", "norm_max", "reverse", "complement"], *args, **kwargs)

        self.instance = None

        is_preset = False

        self._base = dict()
        self._extra = dict() if extra_cols else None

        vals = None

        is_preset = False

        if model is not None: 
            #Initialize from another PoreModel
            if isinstance(getattr(model, "PRMS", None), PoreModelParams):
                self._init(model.PRMS, model)
            
            elif isinstance(model, pd.DataFrame):
                vals = self._vals_from_df(prms, model, normalize)
                self._init_new(prms, *vals)

            elif isinstance(model, dict):
                vals = self._vals_from_dict(prms, model, normalize)
                self._init_new(prms, *vals)

            else:
                raise TypeError(f"Invalid PoreModel type: {type(model)}")
        elif len(prms.name) > 0:
            cache_key = prms.to_key()
            if cache and cache_key in CACHE:
                self._init(prms, CACHE[cache_key])
                return

            if prms.name in self.PRESETS:
                filename = os.path.join(self.PRESET_DIR, prms.name + self.PRESET_EXT)
                ext = self.PRESET_EXT[1:]
            else:
                filename = prms.name
                ext = filename.split(".")[-1]

            if os.path.exists(filename):
                loader = self.FILE_LOADERS.get(ext, PoreModel._vals_from_tsv)
                vals = loader(self, filename, prms)
                self._init_new(prms, *vals)

            else:
                raise FileNotFoundError(f"PoreModel file not found: {filename}.\nSpecify valid filename or preset: {self.PRESETS_STR}")

        else:
            self._init_new(prms)
            
        cache_key = self.PRMS.to_key()
        if cache and not cache_key in CACHE:
            CACHE[cache_key] = self

    def _init_new(self, prms, *args):
        if prms.k < 8:
            ModelType = _uncalled4.PoreModelU16
        else:
            ModelType = _uncalled4.PoreModelU32

        if prms.shift < 0:
            prms.shift = PoreModel.get_kmer_shift(prms.k)

        self._init(prms, ModelType(prms, *args))

        if prms.shift > 0 and self.PRMS.shift != prms.shift:
            self.PRMS.shift = prms.shift,

    def _init(self, prms, model):

        if isinstance(model, PoreModel):
            self.instance = model.instance
            self._base.update(model._base)
            self._extra.update(model._extra)
        else:
            self.instance = model
            self._base["current.mean"] = model.current.mean.to_numpy()
            self._base["current.stdv"] = model.current.stdv.to_numpy()
            self._base["current_sd.mean"] = model.current_sd.mean.to_numpy()
            self._base["current_sd.stdv"] = model.current_sd.stdv.to_numpy()

        self.ModelType = type(self.instance)

        if prms.name is not None:
            self.PRMS.name = prms.name
        if prms.k > 0:
            self.PRMS.k = prms.k
        if prms.shift >= 0:
            self.PRMS.shift = prms.shift

        if self.K >= 8:
            self.kmer_dtype = "uint32"
            self.array_type = ArrayU32
            self.SeqType = _uncalled4.SequenceU32
        else:
            self.kmer_dtype = "uint16"
            self.array_type = ArrayU16
            self.SeqType = _uncalled4.SequenceU16

        self.KMERS = np.arange(self.KMER_COUNT)
        self._KMER_STRS = None

    @property
    def KMER_STRS(self):
        if self._KMER_STRS is None:
            self._KMER_STRS = self.kmer_to_str(self.KMERS)
        return self._KMER_STRS

    @property
    def name(self):
        return self.PRMS.name

    @property
    def kmer_trim(self):
        return (self.PRMS.shift, self.K-self.PRMS.shift-1)

    @property
    def reverse(self):
        return self.PRMS.reverse

    COLUMNS = {"kmer", "current.mean", "current.stdv", "current_sd.mean", "current_sd.stdv"}
    TSV_RENAME = {
        "current" : "current.mean",
        "mean" : "current.mean", 
        "stdv" : "current.stdv",
        "level_mean" : "current.mean", 
        "level_stdv" : "current.stdv",
        "sd_mean"         : "current_sd.mean",
        "sd_stdv"         : "current_sd.stdv",
        "sd"         : "current.stdv",
    }

    def _vals_from_df(self, prms, df, preprocess):
        df = df.rename(columns=self.TSV_RENAME).copy()
        if self._extra is not None:
            extra = df.columns.difference(self.COLUMNS)
            for col in extra:
                #self._cols[col] = df[col].to_numpy()
                self._extra[col] = df[col].to_numpy()

        if "kmer" in df.columns:
            if preprocess:
                df = df.reset_index().sort_values("kmer")

            if prms.k < 0:
                kmer_lens = df["kmer"].str.len().value_counts()
                if len(kmer_lens) > 1:
                    raise ValueError("All kmer lengths must be the same, found lengths: " + ", ".join(map(str, kmer_lens.index)))
                prms.k = kmer_lens.index[0]
        else:
            k = np.log2(len(df))/2.0
            if k != np.round(k):
                raise ValueError("DataFrame length must be power of 4 or include 'kmer' column")
            prms.k = int(k)


        get = lambda c: df[c] if c in df else []

        return (get("current.mean"), get("current.stdv"), get("current_sd.mean"), get("current_sd.stdv"), preprocess)
    
    def _usecol(self, name):
        return self._extra is not None or name in self.COLUMNS or name in self.TSV_RENAME

    def _vals_from_dict(self, prms, d, normalize):
        for name,typ in PARAM_TYPES.items():
            dname = "_"+name
            if dname in d:
                old_val = getattr(prms, name)
                new_val = typ(d[dname])
                if ((typ != np.int32 or new_val >= 0) and 
                    (not hasattr(new_val, "__len__") or len(new_val) > 0)):
                    setattr(prms, name, new_val)
                del d[dname]


        if self._extra is not None:
            for k,v in d.items():
                if k in self.TSV_RENAME:
                    k = self.TSV_RENAME[k]
                if k not in self.COLUMNS:
                    self._extra[k] = v

        get = lambda c: d[c] if c in d else []

        return (get("current.mean"), get("current.stdv"), get("current_sd.mean"), get("current_sd.stdv"), bool(normalize))


    def _vals_from_npz(self, filename, prms, normalize=False):
        d = dict(np.load(filename))
        return self._vals_from_dict(prms, d, normalize)

    def _vals_from_tsv(self, filename, prms, normalize=True):
        df = pd.read_csv(filename, sep=r"\s+", comment="#", usecols=self._usecol)
        return self._vals_from_df(prms, df, normalize)

    def _vals_from_hdf5(self, filename, prms, normalize=True):
        handle = h5py.File(filename, "r")
        df = pd.DataFrame(handle["model"][()])#.reset_index()
        return self._vals_from_df(prms, df, normalize)

    def keys(self):
        return itertools.chain(self._base.keys(), self._extra.keys())

    def __getitem__(self, kmers):
        if isinstance(kmers, str):
            kmers = self.kmer_array(self.str_to_kmers(kmers))
        else:
            kmers = self.kmer_array(kmers)
        seq = self.SeqType(self.instance, kmers)
        return Sequence(seq)

    def str_to_seq(self, bases, fwd=True, name=""):
        coord = RefCoord(name,0,len(bases),fwd)
        return Sequence(self.SeqType(self.instance, bases, coord))


    def __getattr__(self, name):
        ret = getattr(self.instance, name, None)
        if ret is None:
            ret = getattr(self.PRMS, name, None)
            if ret is None:
                raise ValueError(f"PoreModel has no attribute '{name}'")
        return ret #self.instance.__getattribute__(name)

    def kmer_array(self, kmer):
        arr = np.array(kmer)
        if arr.shape == ():
            arr.shape = 1

        if arr.dtype.type in {np.str_, np.bytes_}:
            #TODO add option to fully check BP validity
            if not np.all(np.char.str_len(arr) == self.K):
                raise RuntimeError("All k-mers must be %d bases long" % self.K)
            arr = self.str_to_kmer(kmer)
        v = arr.astype(self.kmer_dtype)
        return self.array_type(v)

    def str_to_kmer(self, kmer):
        fn = lambda k: self.instance.str_to_kmer(k, 0) 
        if isinstance(kmer, (collections.abc.Sequence, np.ndarray, pd.Series, self.array_type, list, tuple)):
            return np.array([self.instance.str_to_kmer(k, 0) for k in kmer])
        return self.instance.str_to_kmer(kmer, 0)
            
    def str_to_kmers(self, seq):
        return self.instance.str_to_kmers(seq).to_numpy()

    def kmer_to_str(self, kmer, dtype=str):
        #, self.ModelType.KmerArray
        if isinstance(kmer, (collections.abc.Sequence, np.ndarray, pd.Series, self.array_type)):
            return self.instance.kmer_to_arr(kmer).astype(dtype)
        return dtype(self.instance.kmer_to_str(kmer))


    def abs_diff(self, current, kmer):
        return self.instance.abs_diff(self, current, self.kmer_array(kmer))

    def to_dict(self, kmer_str=True, params=False):
        if kmer_str:
            d = {"kmer" : self.KMER_STRS}
        else:
            d = {"kmer" : self.KMERS}
        d.update(self._base)
        d.update(self._extra)
        if params:
            d.update(self.params_to_dict("_"))
        return d

    def params_to_dict(self, prefix=""):
        d = dict()
        for name in PARAM_TYPES.keys():
            val = getattr(self.PRMS, name)
            if not isinstance(val, str) or len(val) > 0:
                d[f"{prefix}{name}"] = val 
        return d

    def to_df(self, kmer_str=True, bases=False):
        df = pd.DataFrame({
            key : vals for key,vals in self.to_dict(kmer_str).items()
            if len(vals) > 0
        })

        if bases:
            for b in range(self.K):
                df[b] = self.kmer_base(self.KMERS, b)
        return df
    
    def reduce(self,st,en):
        if st == 0 and en == self.K:
            return self
        elif st < 0 or en > self.K: 
            return None
        bases = list(np.arange(st, en))
        k = len(bases)
        df = self.to_df(bases=True)
        grp = df.groupby(by=bases)
        df = pd.DataFrame({
            "mean" : grp["current.mean"].mean()#.reset_index(drop=True)
        })
        df["kmer"] = np.arange(len(df))
        m = PoreModel(model=df,k=k)
        return m

    def expand_old(self, start):
        k = self.k+1
        kmers = np.arange(4**k)
        if start:
            old_kmers = kmers & ((1 << (2*self.k)) - 1)
        else:
            old_kmers = kmers >> 2

        df = pd.DataFrame({
            "mean" : self.current.mean.to_numpy()[old_kmers],
            "stdv" : self.current.stdv.to_numpy()[old_kmers],
        })
        return PoreModel(model=df,k=5,shift=self.shift+start)


    def expand(self, start=0, end=0):
        k = self.k+start+end
        kmers = np.arange(4**k)
        for i in range(end):
            kmers >>= 2
        kmers &= ((1 << (2*self.k)) - 1)

        df = pd.DataFrame({
            "mean" : self.current.mean.to_numpy()[kmers],
        })
        if len(self.current.stdv) > 0:
            df["stdv"] = self.current.stdv.to_numpy()[kmers]

        p = PoreModelParams(self.PRMS)
        p.k = k
        p.shift += start
        m = PoreModel(model=df)
        m.PRMS = p
        return m

    def to_tsv(self, out=None, header=True):
        if header:
            header = "".join([f"#{n}\t{v}\n" for n,v in self.params_to_dict().items()])
        else:
            header = ""
        tsv = header + self.to_df().to_csv(sep="\t", index=False)

        if out is None:
            sys.stdout.write(tsv)
        elif isinstance(out, str):
            with open(out,"w") as f:
                f.write(tsv)
        else:
            out.write(tsv)

    def to_npz(self, fname):
        np.savez_compressed(fname, **self.to_dict(params=True))

    def norm_mom_params(self, current, tgt_mean=None, tgt_stdv=None):
        tgt_mean = self.model_mean if tgt_mean is None else tgt_mean
        tgt_stdv = self.model_stdv if tgt_stdv is None else tgt_stdv
        scale = tgt_stdv / np.std(current)
        shift = tgt_mean - scale * np.mean(current)
        return scale, shift

    def norm_mad_params(self, current):
        shift = np.median(current)
        scale = 1 / np.median(np.abs(current - shift))
        shift *= -scale
        return scale, shift

    def get_normalized(self, scale, shift):
        means = self.current.mean.to_numpy() * scale + shift
        vals = np.ravel(np.dstack([means,self.stdvs]))
        return PoreModel(self.ModelType(vals), name=self.name)

    def pattern_mask(self, pattern, kmers):
        mask = np.ones(len(kmers), bool)
        for i in range(self.instance.K):
            nucs = NT_CODES[pattern[i]]
            if nucs is None:
                continue
            pos_mask = np.zeros(len(kmers), bool)
            for n in nucs:
                pos_mask |= (self.instance.kmer_base(kmers, i) == n)
            mask &= pos_mask
        return mask

    def pattern_filter(self, pattern, kmers=None):
        if kmers is None:
            kmers = self.KMERS
        return kmers[self.pattern_mask(pattern, kmers)]
    
    def __setstate__(self, d):
        self.__init__(model=d, normalize=False)

    def __getstate__(self):
        d = self.to_dict(params=True)
        return d
    
    def __repr__(self):
        ret = "<PoreModel pa_mean=%.3f pa_stdv=%.3f>\n" % (self.pa_mean, self.pa_stdv)
        ret += str(self.to_df())
        return ret[:-1]
    
    FILE_LOADERS = {
        "npz" : _vals_from_npz,
        "h5" : _vals_from_hdf5,
        "hdf5" : _vals_from_hdf5,
    }

PoreModel._init_presets()

class Sequence:
    LAYERS = {"pos", "mpos", "pac", "name", "fwd", "strand", "kmer", "current", "bases", "base"}
    CONST_LAYERS = {"name", "fwd", "strand"}
    DEFAULT_LAYERS = ["pos", "kmer"]

    def __init__(self, seq, offset=0):
        self.instance = seq
        self.offset = offset
        self.index = self.instance.mpos

    @property
    def name(self):
        return self.coord.name

    @property
    def is_flipped(self):
        return self.index.start < 0

    @property
    def mpos(self):
        return self.index.expand().to_numpy()

    @property
    def pos(self):
        if self.is_flipped:
            return -self.mpos-1
        return self.mpos

    @property
    def pac(self):
        return self.offset + self.pos

    @property
    def strand(self):
        return "+" if self.fwd else "-"

    @property
    def base(self):
        return self.model.kmer_base(self.kmer, self.model.PRMS.shift)

    @property
    def bases(self):
        return self.model.kmer_to_arr(self.kmer).astype(str)

    @property
    def fwd(self):
        return self.is_fwd

    def __len__(self):
        return len(self.instance)

    def _iter_layers(self, names):
        ret = list()

    def to_pandas(self, layers=None, index="mpos"):
        if layers is None:
            layers = ["kmer", "current"]

        cols = dict()
        for name in layers:
            val = getattr(self, name)
            if name in self.CONST_LAYERS:
                val = np.full(len(self), val)
            cols[name] = val
        cols["index"] = getattr(self, index)
        return pd.DataFrame(cols).set_index("index")

    def __getattr__(self, name):
        if not hasattr(self.instance, name):
            raise AttributeError(f"Sequence has no attribute '{name}'")
        return self.instance.__getattribute__(name)
