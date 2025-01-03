from ..signal_processor import ProcessedRead
from ..ref_index import str_to_coord, RefCoord
from ..pore_model import PoreModel
from ..tracks import AlnTrack, AlnDF
from . import TrackIO
import _uncalled4

import numpy as np
import pandas as pd
import sys

class Eventalign(TrackIO):
    FORMAT = "eventalign"

    def __init__(self, filename, write, tracks, track_count):
        TrackIO.__init__(self, filename, write, tracks, track_count)

        self._header = True

        if self.write_mode:
            self.init_write_mode()
        else:
            self.init_read_mode()

    def init_write_mode(self):
        TrackIO.init_write_mode(self)

        flags = set(self.prms.eventalign_flags)
        self.write_read_name = "print-read-names" in flags
        self.write_signal_index = "signal-index" in flags
        self.write_samples = "samples" in flags

        self.header = ["contig", "position", "reference_kmer"]

        if self.write_read_name:
            self.header.append("read_name")
        else:
            self.header.append("read_index")

        self.header += ["strand", "event_index", "event_level_mean", "event_stdv", "event_length", "model_kmer", "model_mean", "model_stdv", "standardized_level"]

        if self.write_signal_index:
            self.header += ["start_idx", "end_idx"]

        if self.write_samples:
            self.header += ["samples"]

        TrackIO._init_output(self, self.prms.buffered)

        if not self.prms.buffered:
            self.output.write("\t".join(self.header) + "\n")

    def init_read_mode(self):
        name = self.filename

        self.output = None
        
        if self.conf.pore_model.name == "r94_dna":
            self.conf.pore_model.name = "r9.4_dna_450bps_6mer_npl"

        self.model = None

        self.track_in = self.init_track(name, name, self.conf)

    #def write_layers(self, track, groups):
    def write_alignment(self, aln):
        events = aln.to_pandas(["seq.kmer", "dtw"], ["seq.pos"]).sort_index().droplevel(0, axis=1)

        model = self.tracks.model
        kmers = events["kmer"]

        std_level = (events["current"] - model.model_mean) / model.model_stdv

        if "events" in events:
            event_counts = events["events"]
            event_index = (event_counts.cumsum() - event_counts.iloc[0]).astype(int)
            if not aln.fwd: #TODO check for flipped ref
                event_index = event_index.max() - event_index
            event_index += 1
        else:
            event_index = np.arange(len(events))

        evts = events.rename(columns={"current" : "mean", "current_sd" : "stdv"}).dropna()
        if self.read is not None:
            self.read.set_events(evts)
            read = self.read
        else:
            read = ProcessedRead(evts)

        if self.write_samples:
            signal = model.norm_to_pa(read.get_norm_signal())
            if len(signal) == 0:
                raise RuntimeError("Failed to output read signal")
        else:
            signal = []

        if self.write_read_name:
            read_id = aln.read_id #track.alignments["read_id"].iloc[0]
        else:
            read_id = str(aln.id)
        self.next_aln_id()
        
        if isinstance(model.instance, _uncalled4.PoreModelU16):
            self.writer = _uncalled4.write_eventalign_new_U16
        elif isinstance(model.instance, _uncalled4.PoreModelU32):
            self.writer = _uncalled4.write_eventalign_new_U32
        else:
            raise ValueError(f"Unknown PoreModel type: {model.instance}")

        eventalign = self.writer(aln.instance, self.write_read_name, self.write_signal_index, signal) #TODO compute internally?

        self._set_output(eventalign)

    def iter_alns(self, layers=None, track_id=None, coords=None, aln_id=None, read_id=None, fwd=None, full_overlap=None, ref_index=None):

        if self.model is None:
            self.model = self.tracks.model

        class SamCache:
            def __init__(self, tracks):
                self.iter = tracks.bam_in.iter_sam()
                self.cache = dict()
                self.next_aln = 0

            def get(self, aln_id):
                while self.next_aln <= aln_id:
                    try:
                        self.cache[self.next_aln] = next(self.iter)
                    except StopIteration:
                        raise RuntimeError(f"Eventalign read_index is greater than BAM file length: {aln_id}")
                    self.next_aln += 1

                ret = self.cache[aln_id]
                del self.cache[aln_id]
                return ret

        sam_cache = SamCache(self.tracks)

        def iter_layers(events):
            groups = events.groupby(["contig", "read_index"])
            for (contig,aln_id), df in groups:

                sam = sam_cache.get(aln_id)

                read_id = sam.query_name

                if self.read_filter is not None and read_id not in self.read_filter:
                    continue

                #filter out skipped positions
                df = df[df["model_mean"] > 0].copy()

                start = df["position"].min()
                end = df["position"].max()
                
                if sam.reference_name != contig or start < sam.reference_start or end > sam.reference_end:
                    raise ValueError(f"Eventalign coordinates ({contig}:{start}-{end}) not contained in corresponding SAM coordinates ({sam.reference_name}:{sam.reference_start}-{sam.reference_end}) for read_index {aln_id}")

                aln = self.tracks.bam_in.sam_to_aln(sam, load_moves=False)
                
                fwd = int( (df["event_index"].iloc[0] < df["event_index"].iloc[-1]) == (df["position"].iloc[0] < df["position"].iloc[-1]))

                df['length'] = (df['event_length'] * self.model.sample_rate).round().astype(int)

                pos = df["position"].to_numpy()
                df["mpos"] = self.tracks.index.pos_to_mpos(pos, fwd)-self.model.shift
                df["mean_cml"]  = df["length"] * df["event_level_mean"]
                df["stdv_cml"]  = df["length"] * df["event_stdv"]

                grp = df.groupby("mpos")

                lengths = grp["length"].sum()
                df = pd.DataFrame({
                    "start" : grp["start_idx"].min(),
                    "length" : lengths,
                    "mean" : self.model.pa_to_norm(grp["mean_cml"].sum() / lengths),
                    "stdv" : self.model.pa_sd_to_norm(grp["stdv_cml"].sum() / lengths)
                }).set_index(grp["mpos"].min())

                df = df.reindex(pd.RangeIndex(df.index.min(), df.index.max()+1))
                df["length"].fillna(-1, inplace=True)

                coords = RefCoord(sam.reference_name, start, end+self.model.K, fwd)
                aln = self.tracks.init_alignment(self.track_in.name, self.next_aln_id(), read_id, sam, start, end+self.model.K)
                dtw = AlnDF(aln.seq, df["start"], df["length"], df["mean"], df["stdv"])
                aln.set_dtw(dtw)
                yield aln

        leftover = pd.DataFrame()

        csv_iter = pd.read_csv(
            self.filename, sep="\t", chunksize=10000,
            usecols=["read_index","contig","position", "event_index",
                     "start_idx","end_idx","event_level_mean","event_stdv","model_mean",
                     "event_length","strand","model_kmer"])

        for events in csv_iter:
            events = pd.concat([leftover, events])

            i = events["read_index"] == events["read_index"].iloc[-1]
            leftover = events[i]
            events = events[~i]

            for aln in iter_layers(events):
                yield aln#s,layers

        if len(leftover) > 0:
            for aln in iter_layers(leftover):
                yield aln#s,layers


    def init_fast5(self, fast5):
        pass

    def init_read(self, read_id, fast5_id):
        pass

    def close(self):
        if not self.prms.buffered and self.output is not None:
            self.output.close()
