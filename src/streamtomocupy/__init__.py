from pkg_resources import get_distribution, DistributionNotFound

__version__ = "0.0.1"

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

from streamtomocupy.config import *
from streamtomocupy.rec import *
from streamtomocupy.streamrecon import *
from streamtomocupy.utils import *
from streamtomocupy.config import *
from streamtomocupy.proc import *

from streamtomocupy.fourierrec import *
from streamtomocupy.find_center import *
