from __future__ import annotations
import math
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString
from .models import Building
from .base import ImageGeoref, MetricProjector, load_geojson_geoms
try:
    from skimage.morphology import skeletonize
except Exception:
    skeletonize=None

def detect_buildings_auto(image_path: Path, georef: ImageGeoref, cfg: dict) -> List[Building]:
    """Conservative classical-CV roof candidate detector; production use should prefer verified GIS/ML polygons."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    max_side = int(cfg.get("work_max_side_px", 2400))
    scale = min(1.0, max_side / max(img.shape[:2]))
    work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(clahe, int(cfg.get("canny_low", 50)), int(cfg.get("canny_high", 140)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mpp = georef.approx_m_per_pixel() / max(scale, 1e-9)
    min_area = float(cfg.get("private_min_area_m2", 30))
    max_area = float(cfg.get("building_max_area_m2", 5000))
    mdu_area = float(cfg.get("mdu_min_area_m2", 350))
    private_max = float(cfg.get("private_max_area_m2", 350))
    min_rect = float(cfg.get("min_rectangularity", 0.35))
    candidates = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px <= 2:
            continue
        area_m2 = area_px * (mpp ** 2)
        if not (min_area <= area_m2 <= max_area):
            continue
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        rect_area = rw * rh
        rectangularity = min(1.0, area_px / max(rect_area, 1e-6))
        if rectangularity < min_rect:
            continue
        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-9:
            continue
        cx = (M["m10"] / M["m00"]) / scale
        cy = (M["m01"] / M["m00"]) / scale
        lon, lat = georef.pixel_to_lonlat(cx, cy)
        aspect = max(rw, rh) / max(1e-6, min(rw, rh))
        conf = 0.35 + 0.35 * rectangularity + 0.15 * min(1.0, area_m2 / 120.0) + 0.15 * (1.0 if aspect < 4.0 else 0.4)
        if area_m2 >= mdu_area and rectangularity >= 0.5 and aspect <= float(cfg.get("mdu_max_aspect", 6.0)):
            kind = "mdu"
        elif area_m2 <= private_max:
            kind = "private"
        else:
            kind = "review"
        if conf < float(cfg.get("min_active_confidence", 0.55)):
            kind = "review"
        candidates.append(Building("", lon, lat, kind, float(min(conf, 0.99)), area_m2, "auto_cv", f"rect={rectangularity:.2f}; aspect={aspect:.2f}"))
    if not candidates:
        return []
    lon0 = float(np.mean([b.lon for b in candidates])); lat0 = float(np.mean([b.lat for b in candidates]))
    proj = MetricProjector(lon0, lat0); min_sep = float(cfg.get("dedupe_distance_m", 8.0)); kept: List[Building] = []
    for b in sorted(candidates, key=lambda z: (-z.confidence, -z.area_m2)):
        x, y = proj.xy(b.lon, b.lat); duplicate = False
        for k in kept:
            kx, ky = proj.xy(k.lon, k.lat)
            if math.hypot(x - kx, y - ky) < min_sep:
                duplicate = True; break
        if not duplicate:
            kept.append(b)
    for i, b in enumerate(kept, 1):
        b.id = f"BLD-{i:05d}"
    return kept

def load_buildings_from_geojson(path: Path) -> List[Building]:
    rows = load_geojson_geoms(path); out: List[Building] = []
    for i, (geom, p) in enumerate(rows, 1):
        c = geom.centroid
        kind = str(p.get("building_type") or p.get("kind") or p.get("type") or "review").lower()
        if "част" in kind or kind == "private_house" or kind == "private":
            kind = "private"
        elif "mdu" in kind or "мкд" in kind or "apartment" in kind:
            kind = "mdu"
        elif kind not in ("private", "mdu", "review"):
            kind = "review"
        out.append(Building(str(p.get("building_id") or p.get("id") or f"BLD-{i:05d}"), float(c.x), float(c.y), kind, float(p.get("confidence", 0.9)), float(p.get("area_m2", 0.0)), "geojson", str(p.get("note", "")), int(p.get("absorbed_households", p.get("households_absorbed", 0)) or 0)))
    return out

def reconcile_building_count(buildings: List[Building], target_private: Optional[int], mode: str) -> List[Building]:
    if not target_private or mode == "none":
        return buildings
    private = [b for b in buildings if b.kind == "private"]
    if len(private) <= target_private:
        return buildings
    ranked = sorted(private, key=lambda b: (-b.confidence, -b.area_m2)); keep = {b.id for b in ranked[:target_private]}
    for b in buildings:
        if b.kind == "private" and b.id not in keep:
            b.kind = "review"; b.note = (b.note + "; count_reconciliation_demoted").strip("; ")
    return buildings

def detect_roads_auto(image_path: Path, georef: ImageGeoref, cfg: dict) -> List[LineString]:
    """Conservative road-centerline baseline; production use should prefer verified road vectors."""
    if skeletonize is None:
        raise RuntimeError("scikit-image is required for auto road extraction")
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    max_side = int(cfg.get("work_max_side_px", 1800)); scale = min(1.0, max_side / max(img.shape[:2]))
    work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV); H, S, V = cv2.split(hsv)
    mask = ((S < int(cfg.get("road_sat_max", 80))) & (V > int(cfg.get("road_val_min", 85)))).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)); mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1); mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    skel = skeletonize(mask > 0); ys, xs = np.nonzero(skel); pix = set(zip(xs.tolist(), ys.tolist()))
    if not pix:
        return []
    neigh8 = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]; deg = {p:sum(((p[0]+dx,p[1]+dy) in pix) for dx,dy in neigh8) for p in pix}; anchors = {p for p,d in deg.items() if d != 2}; visited_edges=set(); lines=[]
    def edge_key(a,b): return tuple(sorted((a,b)))
    for a in list(anchors):
        for dx,dy in neigh8:
            b=(a[0]+dx,a[1]+dy)
            if b not in pix or edge_key(a,b) in visited_edges: continue
            path=[a,b]; visited_edges.add(edge_key(a,b)); prev,cur=a,b
            while cur not in anchors:
                nxts=[]
                for ddx,ddy in neigh8:
                    q=(cur[0]+ddx,cur[1]+ddy)
                    if q in pix and q != prev and edge_key(cur,q) not in visited_edges: nxts.append(q)
                if not nxts: break
                q=nxts[0]; visited_edges.add(edge_key(cur,q)); path.append(q); prev,cur=cur,q
            if len(path) < int(cfg.get("road_min_trace_pixels", 8)): continue
            coords=[]
            for x,y in path[::max(1,int(cfg.get("road_trace_stride",3)))]:
                lon,lat=georef.pixel_to_lonlat(x/scale,y/scale); coords.append((lon,lat))
            if len(coords)>=2: lines.append(LineString(coords))
    return lines

def load_roads_from_geojson(path: Path) -> List[LineString]:
    out=[]
    for geom, p in load_geojson_geoms(path):
        if isinstance(geom, LineString): out.append(geom)
        elif isinstance(geom, MultiLineString): out.extend(list(geom.geoms))
    return out
