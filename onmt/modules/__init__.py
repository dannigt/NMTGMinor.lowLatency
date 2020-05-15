from onmt.modules.GlobalAttention import GlobalAttention
from onmt.modules.ImageEncoder import ImageEncoder
from onmt.modules.BaseModel import Generator, NMTModel
from onmt.modules.StaticDropout import StaticDropout

# For flake8 compatibility.
__all__ = [GlobalAttention, ImageEncoder, Generator, NMTModel, StaticDropout]
