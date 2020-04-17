from .EncoderBottleNeckHead import EncodeDecodeBase
try:
    from myTorch import nn, Models
except ImportError:
    from .. import nn, Models
import torch


class PSPNetBuilder(EncodeDecodeBase)