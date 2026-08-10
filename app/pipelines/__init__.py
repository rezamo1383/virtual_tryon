"""Domain pipelines exposed to the task router."""

from app.pipelines.base import BasePipeline
from app.pipelines.clothing import ClothingPipeline
from app.pipelines.wallpaper import WallpaperPipeline

__all__ = ["BasePipeline", "ClothingPipeline", "WallpaperPipeline"]
