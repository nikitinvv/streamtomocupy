import argparse
import configparser
from collections import OrderedDict

SECTIONS = OrderedDict()

SECTIONS["reconstruction"] = {
    "file-type": {
        "default": "standard",
        "type": str,
        "help": "Input file type",
        "choices": ["standard", "double_fov"],
    },
    "reconstruction-algorithm": {
        "default": "fourierrec",
        "type": str,
        "help": "Reconstruction algorithm",
        "choices": ["fourierrec", "lprec", "linerec"],
    },
    "rotation-axis": {
        "default": -1.0,
        "type": float,
        "help": "Location of rotation axis",
    },
    "dtype": {
        "default": "float32",
        "type": str,
        "choices": ["float32", "float16"],
        "help": "Data type used for reconstruction. Note float16 works with power of 2 sizes.",
    },
    "fbp-filter": {
        "default": "parzen",
        "type": str,
        "help": "Filter for FBP reconstruction",
        "choices": [
            "none",
            "ramp",
            "shepp",
            "hann",
            "hamming",
            "parzen",
            "cosine",
            "cosine2",
        ],
    },
    "dezinger": {
        "type": int,
        "default": 0,
        "help": "Width of region for removing outliers",
    },
    "dezinger-threshold": {
        "type": int,
        "default": 5000,
        "help": "Threshold of grayscale above local median to be considered a zinger pixel",
    },
    "minus-log": {"default": "True", "help": "Take -log or not"},
    "nsino-per-chunk": {
        "type": int,
        "default": 8,
        "help": "Number of sinograms per chunk. Use larger numbers with computers with larger memory. ",
    },
    "nproj-per-chunk": {
        "type": int,
        "default": 8,
        "help": "Number of projections per chunk. Use larger numbers with computers with larger memory.  ",
    },
    "ngpus": {
        "type": int,
        "default": 1,
        "help": "Number of GPUs for processing.  ",
    },
}

SECTIONS["retrieve-phase"] = {
    "retrieve-phase-method": {
        "default": "none",
        "type": str,
        "help": "Phase retrieval correction method",
        "choices": ["none", "paganin", "Gpaganin"],
    },
    "energy": {"default": 20, "type": float, "help": "X-ray energy [keV]"},
    "propagation-distance": {
        "default": 100,
        "type": float,
        "help": "Sample detector distance [mm]",
    },
    "retrieve-phase-alpha": {
        "default": 0.001,
        "type": float,
        "help": "Regularization parameter",
    },
    "retrieve-phase-delta-beta": {
        "default": 1500.0,
        "type": float,
        "help": "delta/beta material for Generalized Paganin",
    },
    "retrieve-phase-W": {
        "default": 2e-4,
        "type": float,
        "help": "Characteristic transverse length for Generalized Paganin",
    },
    "retrieve-phase-pad": {
        "type": int,
        "default": 1,
        "help": "Padding with extra slices in z for phase-retrieval filtering",
    },
    "pixel-size": {"default": 1, "type": float, "help": "Pixel size [microns]"},
}

SECTIONS["remove-stripe"] = {
    "remove-stripe-method": {
        "default": "none",
        "type": str,
        "help": "Remove stripe method: none, fourier-wavelet, titarenko",
        "choices": ["none", "fw", "ti", "vo-all"],
    },
}

SECTIONS["fw"] = {
    "fw-sigma": {
        "default": 1,
        "type": float,
        "help": "Fourier-Wavelet remove stripe damping parameter",
    },
    "fw-filter": {
        "default": "sym16",
        "type": str,
        "help": "Fourier-Wavelet remove stripe filter",
        "choices": ["haar", "db5", "sym5", "sym16"],
    },
    "fw-level": {
        "type": int,
        "default": 7,
        "help": "Fourier-Wavelet remove stripe level parameter",
    },
    "fw-pad": {
        "default": True,
        "help": "When set, Fourier-Wavelet remove stripe extend the size of the sinogram by padding with zeros",
        "action": "store_true",
    },
}

SECTIONS["vo-all"] = {
    "vo-all-snr": {
        "default": 3,
        "type": float,
        "help": "Ratio used to locate large stripes. Greater is less sensitive.",
    },
    "vo-all-la-size": {
        "default": 61,
        "type": int,
        "help": "Window size of the median filter to remove large stripes.",
    },
    "vo-all-sm-size": {
        "type": int,
        "default": 21,
        "help": "Window size of the median filter to remove small-to-medium stripes.",
    },
    "vo-all-dim": {"default": 1, "help": "Dimension of the window."},
}

SECTIONS["ti"] = {
    "ti-beta": {
        "default": 0.022,  # as in the paper
        "type": float,
        "help": "Parameter for ring removal (0,1)",
    },
    "ti-mask": {
        "default": 1,
        "type": float,
        "help": "Mask size for ring removal (0,1)",
    },
}

RECON_STEPS_PARAMS = (
    "reconstruction",
    "remove-stripe",
    "retrieve-phase",
    "fw",
    "ti",
    "vo-all",
)


class Params(object):
    def __init__(self, sections=()):
        self.sections = sections

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument("--{}".format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)
        return parser.parse_args("")


def config_to_list(config_name):
    """
    Read arguments from config file and convert them to a list of keys.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()
    if not config.read([config_name]):
        return []
    for section in SECTIONS:
        for name, opts in (
            (n, o)
            for n, o in SECTIONS[section].items()
            if config.has_option(section, n)
        ):
            value = config.get(section, name)
            if value != "" and value != "None":
                action = opts.get("action", None)
                if action == "store_true" and value == "True":
                    result.append("--{}".format(name))
                if not action == "store_true":
                    if opts.get("nargs", None) == "+":
                        result.append("--{}".format(name))
                        result.extend((v.strip() for v in value.split(",")))
                    else:
                        result.append("--{}={}".format(name, value))
    return result


def write_args(config_file, args=None):
    """
    Write *config_file*
    """
    config = configparser.ConfigParser()
    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and hasattr(args, name.replace("-", "_")):
                value = getattr(args, name.replace("-", "_"))
                if isinstance(value, list):
                    value = ", ".join(value)
            else:
                value = opts["default"] if opts["default"] is not None else ""
            prefix = "# " if value == "" else ""
            config.set(section, prefix + name, str(value))
    with open(config_file, "w") as f:
        config.write(f)


def read_args(config_file):
    """
    Read *config_file*
    """
    parser = argparse.ArgumentParser()
    tomo_steps_params = RECON_STEPS_PARAMS
    subparsers = parser.add_subparsers(title="Commands", metavar="")
    cmd_params = Params(sections=tomo_steps_params)
    cmd_parser = subparsers.add_parser(
        "recon_steps", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    cmd_parser = cmd_params.add_arguments(cmd_parser)
    values = ["recon_steps"] + config_to_list(config_name=config_file)
    args = parser.parse_known_args(values)[0]
    return args
