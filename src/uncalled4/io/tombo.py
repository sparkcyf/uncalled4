from ..signal_processor import ProcessedRead
from ..ref_index import str_to_coord, RefCoord
from ..pore_model import PoreModel
from ..tracks import AlnDF
from ..read_index import Fast5Reader
from ..moves import sam_to_ref_moves
from . import TrackIO

import _uncalled4

import numpy as np
import pandas as pd
import sys, os
from ont_fast5_api.fast5_interface import get_fast5_file

class Tombo(TrackIO):
    FORMAT = "tombo"

    def __init__(self, filename, write, tracks, track_count):
        TrackIO.__init__(self, filename, write, tracks, track_count)

        self._header = True

        self.fast5_in = None
        self.read_index = tracks.read_index
        self.read_id_in = None

        if self.write_mode:
            self.init_write_mode()
        else:
            self.init_read_mode()

    def init_write_mode(self):
        raise RuntimeError("Writing in tombo format not supported")

    def init_read_mode(self):
        name = self.filename
        
        self.conf.read_index.paths = self.prms.tombo_in

        self.track_in = self.init_track(name, name, self.conf)

    def iter_alns(self, track_id=None, coords=None, aln_id=None, read_id=None, fwd=None, full_overlap=None, ref_index=None):

        
        read_filter = self.read_index.read_filter #f5reader.get_read_filter()
        paths = self.read_index.file_info.values() #f5reader.prms.paths

        if self.conf.read_index.read_count is not None and len(paths) > self.conf.read_index.read_count:
            paths = paths[:self.conf.read_index.read_count]

        aln_id = 1

        for _,fast5_fname in paths:
            fast5_basename = os.path.basename(fast5_fname)

            try:
                reader = Fast5Reader(fast5_fname)
                fast5 = reader.infile #get_fast5_file(fast5_fname, mode="r")
            except:
                sys.stderr.write(f"Unable to open \"{fast5_fname}\". Skipping.\n")
                continue

            read, = fast5.get_reads()
            handle = read.handle
            read = reader[read.read_id]

            if not (read_filter is None or read.id in read_filter): 
                continue

            if not 'BaseCalled_template' in handle['Analyses']['RawGenomeCorrected_000']:
                continue

            handle = handle['Analyses']['RawGenomeCorrected_000']['BaseCalled_template']
            attrs = handle.attrs
            if attrs["status"] != "success":
                continue

            is_rna = handle.attrs["rna"]
            if is_rna != self.conf.is_rna:
                raise RuntimeError("Reads appear to be RNA but --rna not specified")

            self.fast5_in = fast5_fname
            read_id = read.id

            aln_attrs = dict(handle["Alignment"].attrs)

            chrom = aln_attrs["mapped_chrom"]
            model = self.track_in.model

            shift_st = model.shift
            shift_en = model.K - model.shift - 1

            fwd = aln_attrs["mapped_strand"] == "+"
            sig_fwd = fwd != is_rna
            if sig_fwd:
                start = aln_attrs["mapped_start"]-shift_st-1
                end = aln_attrs["mapped_end"]+shift_en-1
            else:
                start = aln_attrs["mapped_start"]-shift_st+1
                end = aln_attrs["mapped_end"]+shift_en+1

            if start < 0:
                clip = -start
                start = 0
            else:
                clip = 0

            sam = None
            best = 0
            for read_sam in self.tracks.bam_in.iter_read(read_id):
                if read_sam.reference_name == chrom:
                    overlap = max(read_sam.reference_start, start) < min(read_sam.reference_end, end)
                    if best < overlap:
                        best = overlap
                        sam = read_sam

            if sam is None:
                sys.stderr.write(f"Failed to convert {read_id}\n")
                continue

            tombo_events = np.array(handle["Events"])[clip:]

            tombo_start = handle["Events"].attrs["read_start_rel_to_raw"]
            
            raw_len = len(read.signal)
            starts = tombo_events["start"]


            lengths = tombo_events["length"]
            currents = self.track_in.model.pa_to_norm(tombo_events["norm_mean"])

            if is_rna:
                starts = raw_len - tombo_start - starts - tombo_events["length"]
                step = -1
            else:
                starts = tombo_start + starts
                step = 1

            aln = self.tracks.init_alignment(self.track_in.name, self.next_aln_id(), read, sam, start, end)

            dtw = AlnDF(aln.seq, np.array(starts[::step]), np.array(lengths[::step]), np.array(currents[::step])) #df["stdv"])

            aln.set_dtw(dtw)


            moves = sam_to_ref_moves(self.conf, self.tracks.index, read, sam)
            if moves is not None:
                i = max(0, moves.index.start - aln.seq.index.start)
                j = min(len(moves), len(moves) + moves.index.end -  aln.seq.index.end)
                    
                aln.set_moves(moves)

            

            aln_id += 1

            yield aln#s,layers
            
    def write_alignment(self, alns):
        pass


    def init_fast5(self, fast5):
        pass

    def init_read(self, read_id, fast5_id):
        pass

    def close(self):
        pass
