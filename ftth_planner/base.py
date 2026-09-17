from __future__ import annotations
import csv, hashlib, json, math, os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, List
from PIL import Image, ImageFont
import yaml
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
try:
    import rasterio
except Exception:
    rasterio=None

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def feature_collection(features: Sequence[dict]) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}

def round_up_standard(value: float, standards: Sequence[int]) -> int:
    for s in sorted(standards):
        if value <= s:
            return int(s)
    return int(max(standards))

def safe_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

class MetricProjector:
    """Auto-select local UTM for metric calculations."""

    def __init__(self, lon0: float, lat0: float):
        zone = int((lon0 + 180.0) / 6.0) + 1
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        self.crs_geo = CRS.from_epsg(4326)
        self.crs_metric = CRS.from_epsg(epsg)
        self.to_metric = Transformer.from_crs(self.crs_geo, self.crs_metric, always_xy=True)
        self.to_geo = Transformer.from_crs(self.crs_metric, self.crs_geo, always_xy=True)

    def xy(self, lon: float, lat: float) -> Tuple[float, float]:
        return self.to_metric.transform(lon, lat)

    def ll(self, x: float, y: float) -> Tuple[float, float]:
        return self.to_geo.transform(x, y)

    def geom_to_metric(self, geom):
        return shp_transform(self.to_metric.transform, geom)

    def geom_to_geo(self, geom):
        return shp_transform(self.to_geo.transform, geom)

class ImageGeoref:
    """GeoTIFF transform or explicit lon/lat bbox for PNG/JPG."""

    def __init__(self, image_path: Path, config: dict):
        self.image_path = image_path
        self.mode = config.get("mode", "bbox")
        with Image.open(image_path) as im:
            self.width, self.height = im.size
        self.bounds = None
        self._rio_transform = None
        self._rio_crs = None
        if self.mode == "geotiff":
            if rasterio is None:
                raise RuntimeError("rasterio is required for geotiff mode")
            with rasterio.open(image_path) as ds:
                self._rio_transform = ds.transform
                self._rio_crs = ds.crs
                self.bounds = ds.bounds
                if ds.crs is None:
                    raise ValueError("GeoTIFF has no CRS")
                self._to_geo = Transformer.from_crs(ds.crs, CRS.from_epsg(4326), always_xy=True)
                self._from_geo = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
        elif self.mode == "bbox":
            bbox = config.get("bbox_lonlat")
            if not bbox or len(bbox) != 4:
                raise ValueError("georef.bbox_lonlat = [min_lon,min_lat,max_lon,max_lat] is required for bbox mode")
            self.bounds = tuple(float(x) for x in bbox)
        else:
            raise ValueError(f"Unsupported georef.mode: {self.mode}")

    def pixel_to_lonlat(self, px: float, py: float) -> Tuple[float, float]:
        if self.mode == "bbox":
            min_lon, min_lat, max_lon, max_lat = self.bounds
            lon = min_lon + (px / max(1, self.width - 1)) * (max_lon - min_lon)
            lat = max_lat - (py / max(1, self.height - 1)) * (max_lat - min_lat)
            return lon, lat
        x, y = self._rio_transform * (px, py)
        return self._to_geo.transform(x, y)

    def lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        if self.mode == "bbox":
            min_lon, min_lat, max_lon, max_lat = self.bounds
            px = (lon - min_lon) / max(1e-12, (max_lon - min_lon)) * (self.width - 1)
            py = (max_lat - lat) / max(1e-12, (max_lat - min_lat)) * (self.height - 1)
            return px, py
        x, y = self._from_geo.transform(lon, lat)
        inv = ~self._rio_transform
        px, py = inv * (x, y)
        return px, py

    def approx_m_per_pixel(self) -> float:
        cx, cy = self.width / 2, self.height / 2
        lon1, lat1 = self.pixel_to_lonlat(cx, cy)
        lon2, lat2 = self.pixel_to_lonlat(min(self.width - 1, cx + 100), cy)
        p = MetricProjector(lon1, lat1)
        x1, y1 = p.xy(lon1, lat1)
        x2, y2 = p.xy(lon2, lat2)
        return math.hypot(x2 - x1, y2 - y1) / 100.0

def load_geojson_geoms(path: Path, allowed_types: Optional[Sequence[str]] = None) -> List[Tuple[Any, dict]]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    out = []
    for ft in gj.get("features", []):
        props = ft.get("properties", {}) or {}
        if allowed_types and props.get("type") not in allowed_types:
            continue
        out.append((shape(ft["geometry"]), props))
    return out
