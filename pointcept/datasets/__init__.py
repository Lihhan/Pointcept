from .defaults import DefaultDataset, ConcatDataset
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

# indoor scene
from .s3dis import S3DISDataset
from .scannet import ScanNetDataset, ScanNet200Dataset
from .scannetpp import ScanNetPPDataset
from .scannet_pair import ScanNetPairDataset
from .hm3d import HM3DDataset
from .structure3d import Structured3DDataset
from .aeo import AEODataset

# outdoor scene
from .semantic_kitti import SemanticKITTIDataset
from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset

# object
try:
    from .modelnet import ModelNetDataset
except (ImportError, ModuleNotFoundError):
    ModelNetDataset = None

try:
    from .shapenet_part import ShapeNetPartDataset
except (ImportError, ModuleNotFoundError):
    ShapeNetPartDataset = None

# dataloader
from .dataloader import MultiDatasetDataloader
