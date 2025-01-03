from . import config
from . import RefCoord

eventalign_flags = "\", \"".join(["print-read-names", "signal-index", "samples"])#, "scale-events"]

BASECALLER_PROFILES = {
   "dna_r10.4.1_260bps" : 2,
   "dna_r10.4.1_400bps" : 1,
   "dna_r9.4.1_400bps" : 3,
   "rna_r9.4.1_70bps" : 2,
   "rna004_130bps" : 3,
}
BASECALLER_PROFILES_STR = "', '".join(BASECALLER_PROFILES.keys())

class IOParams(config.ParamGroup):
    _name = "io"
IOParams._def_params(
    ("processes", 1, int, "Number of parallel processes"),
    ("bam_chunksize", 500, int, "Per-process alignment bam_chunksize"),

    ("bed_filter", None, str, "Only parse regions in BED file"),

    ("tsv_out", None, str, "TSV output file (or \"-\"/no argument for stdout)"),
    ("tsv_cols", ["dtw"], list, "TSV file output alignment layers (comma-separated, can also include \"read_id\""),
    ("tsv_na", "*", str, "Missing value representation for TSV output"),
    ("tsv_noref", False, bool, "Will NOT output reference coordinates to TSV if True"),

    ("bam_in", None, None, "BAM input file"),
    ("bam_out", None, str, "BAM output file"),
    ("bam_f5c", None, bool, "Include f5c ss: and si: BAM output tags"),

    ("m6anet_out", None, str, "Output m6anet dataprep directory"),

    ("model_dir", None, str, "Pore model training output directory"),

    ("buffered", False, bool, "Will store alignments in buffer rather than directly output"),
    ("bam_header", None, None, "BAM input header dict"),

    ("eventalign_in", None, list, "Eventalign (nanopolish) input file (or \"-\"/no argument for stdin)"),
    ("eventalign_out", None, str, "Eventalign (nanopolish) output file (or \"-\"/no argument for stdout)"),
    ("eventalign_index", None, str, "Nanopolish index file"),
    ("eventalign_flags", [], list, f"Eventalign optional flags (comma-separated list of \"\"{eventalign_flags}\")"),
    ("tombo_in", None, None, "Fast5 files containing Tombo alignments"),

    ("guppy_in", None, str, "Guppy directory containing sequencing summary, BAM files, and FAST5s"),
    ("bam_ss", None, bool, "Output f5c-style ss: tag to BAM format"),

    ("out_name", None, str, "Output track name (defaults to file basename)"),
    ("input_names", None, list, "Names of tracks to read from input(s)"),

    ("init_track", True, bool, "If true will initialze track"),

    ("ordered_out", False, bool,  "Output alignments in the same order as BAM input (default if processes=1)"),
    ("overwrite", False, bool, "Overwrite existing tracks"),
    ("append", False, bool, "Append reads to existing tracks"),
    ("aln_chunksize", 500, int, "Number of alignments to query for iteration"),
    ("ref_chunksize", 10000, int, "Number of reference coordinates to query for iteration"),
    ignore_toml={"bam_in", "bam_out", "sql_in", "sql_out", "eventalign_in", "eventalign_out", "tombo_in", "eventalign_index", "overwrite", "append"},
    config_add=False
)

class TracksParams(config.ParamGroup):
    _name = "tracks"
TracksParams._def_params(
    ("ref", None, str, "Reference FASTA file, must match --bam-in reference"),
    ("self", False, bool, "Perform reference-free signal-to-basecall alignment"),
    ("io", {}, IOParams, "Track input/output parameters"),
    ("ref_bounds", None, RefCoord, "Only load reads which overlap these coordinates"),
    ("read_filter", None, None, "Only load reads which overlap these coordinates"),
    ("max_reads", None, int, "Maximum number of reads to load"),

    ("count_events", False, bool, "Compute and output per-reference event counts, where N > 1 indicates N-1 stays and N < 1 indicates 1/N skips"),
    ("mask_skips", None, None, "Either \"all\" to mask all skips, or \"keep_best\" to mask all but the closest to the model"),
    ("skip_stdv_thresh", None, float, "Either \"all\" to mask all skips, or \"keep_best\" to mask all but the closest to the model"),
    ("mask_indels", None, int, "Mask positions which overlap basecalled alignment insertions or deletions of this length or longer"),
    ("mvcmp_mask", None, float, "Will filter out positions with mvcmp.dist >= mvcmp_mask if specified"),
    ("max_norm_dist", 2, float, "Maximum mvcmp.dist for posititions to be used for iterative normalization"),
    ("max_sd", 1, float, "Maximum current standard deviation"),
    ("min_aln_length", 100, int, "Minimum number of aligned bases"),
    ("basecaller_profile", None, str, f"Basecalling model profile name ('{BASECALLER_PROFILES_STR}')"),

    ("full_overlap", False, bool, "If true will only include reads which fully cover reference bounds"),
    ("min_coverage", 1, int, "Reference positions with less than this coverage will be excluded from each track (or all tracks if shared_refs_only is true)"),
    ("shared_reads_only", False, bool, "If true will only contain reads shared between all tracks"),
    ("shared_refs_only", False, bool, "If true will only contain reference positions where all tracks have sufficient coverage (see min_coverage)"),

    ("layers", [], None, "Layers to load (e.g. current, dwell, model_diff)"),

    ("load_mat", False, bool, "If true will pivot layers into a matrix"), #TODO change to mat_layers, only do it for them

    ("refstats", None, None, "Per-reference summary statistics to compute for each layer"),
    ("refstats_layers", None, None, "Layers to compute refstats"),

    ("load_fast5s", False, bool, "Load fast5 files"),
    ("zero_ts", False, bool, "Set 'ts' BAM tag to 0 (Dorado 0.5.3 bug workaround)"),

    ignore_toml={"ref_bounds", "full_overlap", "refstats", "refstats_layers", "read_filter", "load_fast5s"}
)

class TrainParams(config.ParamGroup):
    _name = "train"
TrainParams._def_params(
    ("kmer_samples", 500, int, "Maximum number of instances of each k-mer to use per training iteration"),
    ("init_model", "", str, "Initial pore model. If not specified, iteration will be based on basecaller move alignments"),
    ("init_mode", "moves_avg", str, "How to initialize pore model if --init-model not specified ('moves_avg', 'moves')"),
    ("train_layers", [], None, "Layers to store during training (default: dtw.current,dtw.current_sd,dtw.length)"),
    ("moves_avg", None, None, "K-mer base offset(s) to average for initial moves-based model (comma seperated)"),
    ("init_events", 100000, int, "Number of events to use for computing picoamp scaling parameters for new pore model"),
    ("kmer_len", None, int, "Output model k-mer length. Required if init_model is not specified"),
    ("iterations", 1, int, "Number of model training iterations"),
    ("buffer_size", 256, int, "Size of sorted chunk buffer (MB)"),
    ("max_moves_dist", 1, float, "Maximum mean_refi_dist from basecalled alignment to use for model trainer"),
    ("train_mean", False, bool, "Compute mean k-mer current levels rather than median"),
    ("skip_dtw", None, bool, "Will use previous training data to re-compute the model. '--out-dir' must be a previous model training directory with at least the specified number of iterations"),
    ("append", False, bool, "If output directory exists and contains a file 'it[N].model.tsv', will use the file with the highest N to initialize training and generate additional training iterations"),
)

class ReadIndexParams(config.ParamGroup):
    _name = "read_index"
ReadIndexParams._def_params(
    ("paths", None, list, "Paths to FAST5, SLOW5, or POD5 files, or to directories containing those files (optionally recursive)"),
    ("read_filter", None, None, "List of read IDs to load, or file containing one read ID per line"),
    ("read_index", None, str, "File containing a mapping of read IDs to filenames"),
    ("default_read_index", "read_index.txt", str, "Filename for auto-generated read-to-file index"),
    ("recursive", None, bool, "Recursively search 'paths' for FAST5, SLOW5, or POD5 files"),
    ("read_count", None, int, "Maximum number of reads to load"),
    ("load_signal", True, bool, "Must be set to true to load signal from FAST5/SLOW5/POD5"),
)

class VisParams(config.ParamGroup):
    _name = "vis"
VisParams._def_params(
    ("track_colors", ["#721ea9", "#4DA91E", "#cc114f", "#4676FF"], list, "Track Colors"),
    ("base_colors", ["#80ff80", "#6b93ff", "#ffe543", "#ff8080"], list, "Colors for each base (A,C,G,T/U)"), 
    ("svg", False, bool, "Make SVG-friendly figures"), 
)

class DotplotParams(config.ParamGroup):
    _name = "dotplot"
DotplotParams._def_params(
    ("tracks", None, None, "DTW aligment tracks"),
    ("moves_track", None, str, "Only display basecalled alignments from this track"),
    ("moves_error", False, bool, "Display basecalled alignment errors"),
    ("show_legend", True, bool, "Display legend"),
    ("show_bands", True, bool, "Display DTW bands (if present in DB)"),
    ("select_ref", None, int, "Display a horizontal line at specified reference coordinate"),
    ("layers", [], None, ""),
)

class SigplotParams(config.ParamGroup):
    _name = "sigplot"
SigplotParams._def_params(
    ("tracks", None, None, "DTW aligment tracks"),
    ("max_reads", 10, int, ""),
    ("show_events", False, bool, "Display event means plotted over signal instead of model current (currently events are computed from scratch)"),
    ("no_model", False, bool, "Will not plot the expected reference signal if True"),
    ("multi_background", False, bool, "Will plot multiple stacked background colors for multiple tracks if True"),
    ("yaxis_fixed", False, bool, ""),
    ("fill_layer", "base", str, ""),
    ("fill_track", 0, None, "")
)

class RefplotParams(config.ParamGroup):
    _name = "refplot"
RefplotParams._def_params(
    ("tracks", None, None, "DTW aligment tracks"),
    ("layer", "current", str, ""),
    ("plot", None, str, "Type of plot to display (hist, violin, box)"),
    ("kmer_coord", None, int, "Center position of k-mer to display"),
)

class TrackplotParams(config.ParamGroup):
    _name = "trackplot"
TrackplotParams._def_params(
    ("tracks", None, None, "DTW aligment tracks"),
    ("panels", None, None, "List of tuples specifying which panels to display. First element of each tuple specifies plot type (mat, box, line, scatter), second element specifies which layer(s) to display."),
    ("select_ref", None, str, "Reference Selection"),
    ("select_read", None, str, "Read Selection"),
    ("hover_read", False, bool, "If True will display read_id in mat hover"),
    ("show_legend", True, bool, "If True will display legend"),
    ("share_reads", False, bool, "If True will only display reads shared by all alignment tracks with shared y-axis"),
    ("width", None, int, "Figure width"),
    ("panel_heights", None, None, "Relative height of each panel"),
    ("min_height", 200, int, "Minimum figure height"),
    ("outfile", None, str, "Output file"),
)

class ReadstatsParams(config.ParamGroup):
    _name = "readstats"
ReadstatsParams._def_params(
    ("layers", None, None, "Which layers to compute statistics"),
    ("stats", None, None, "Summary statistics to compute (any builtin numpy function, e.g. mean, std, etc)"),
    ("pca_layer", "current", str, "Which statistics to use for PCA"),
    ("pca_components", 2, int, "Number of principle components to output for the \"pca\" command."),
    ("summary_stats", ["mean"], None, "Summary statistics to compute for \"model_diff\" command."),
)

