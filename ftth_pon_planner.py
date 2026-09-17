#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable deterministic FTTH/PON planner.

Pipeline: image -> buildings -> roads -> OLT -> PON/FAT/mufta -> road routing ->
optical budget -> deterministic map -> GeoJSON/CSV/QA.

Engineering rule used here: one fusion splice joins two fibers; if an incoming
cable is fully spliced, closure splice capacity equals that cable's fiber count.
"""
from __future__ import annotations
import argparse, csv, json, math, os, urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, MultiLineString, shape, mapping
from shapely.ops import transform as shp_transform

try:
    import rasterio
except Exception:
    rasterio = None

# ---------------- data ----------------
@dataclass
class Building:
    id: str; lon: float; lat: float; kind: str; confidence: float
    area_m2: float = 0.0; note: str = ""

@dataclass
class Fat:
    id: str; pon: str; splitter: str; capacity: int; used: int
    road_node: str; lon: float; lat: float

@dataclass
class Mufta:
    id: str; pon: str; road_node: str; lon: float; lat: float
    incoming_f: int; capacity_splices: int; fat_branches: int


def load_yaml(p):
    with open(p, encoding="utf-8") as f: return yaml.safe_load(f)

def dump_json(p, obj):
    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def dump_csv(p, rows, fields):
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader();
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def fc(features): return {"type":"FeatureCollection","features":features}

def std_capacity(v, standards):
    for s in sorted(standards):
        if v <= s: return int(s)
    return int(max(standards))

# ---------------- image / georef ----------------
def local_image(src: str, work: Path) -> Path:
    if src.startswith("http://") or src.startswith("https://"):
        p=work/"source_image"; urllib.request.urlretrieve(src,p); return p
    return Path(src)

class Georef:
    def __init__(self, image: Path, cfg: dict):
        with Image.open(image) as im: self.w,self.h=im.size
        self.mode=cfg.get("mode","bbox")
        if self.mode=="bbox":
            self.bbox=tuple(map(float,cfg["bbox_lonlat"]))
        elif self.mode=="geotiff":
            if rasterio is None: raise RuntimeError("rasterio required for geotiff")
            self.ds=rasterio.open(image); self.tr=self.ds.transform
            self.to_geo=Transformer.from_crs(self.ds.crs,4326,always_xy=True)
            self.from_geo=Transformer.from_crs(4326,self.ds.crs,always_xy=True)
        else: raise ValueError("georef.mode must be bbox or geotiff")
    def px2ll(self,x,y):
        if self.mode=="bbox":
            a,b,c,d=self.bbox; return a+x/(self.w-1)*(c-a), d-y/(self.h-1)*(d-b)
        X,Y=self.tr*(x,y); return self.to_geo.transform(X,Y)
    def ll2px(self,lon,lat):
        if self.mode=="bbox":
            a,b,c,d=self.bbox; return (lon-a)/(c-a)*(self.w-1),(d-lat)/(d-b)*(self.h-1)
        X,Y=self.from_geo.transform(lon,lat); return (~self.tr)*(X,Y)
    def mpp(self):
        lon,lat=self.px2ll(self.w/2,self.h/2); lon2,lat2=self.px2ll(min(self.w-1,self.w/2+100),self.h/2)
        p=Metric(lon,lat); x1,y1=p.xy(lon,lat); x2,y2=p.xy(lon2,lat2); return math.hypot(x2-x1,y2-y1)/100

class Metric:
    def __init__(self, lon, lat):
        z=int((lon+180)//6)+1; epsg=(32600 if lat>=0 else 32700)+z
        self.fwd=Transformer.from_crs(4326,epsg,always_xy=True); self.rev=Transformer.from_crs(epsg,4326,always_xy=True)
    def xy(self,lon,lat): return self.fwd.transform(lon,lat)
    def ll(self,x,y): return self.rev.transform(x,y)
    def geom(self,g): return shp_transform(self.fwd.transform,g)

# ---------------- buildings ----------------
def buildings_geojson(path: Path):
    out=[]
    gj=json.load(open(path,encoding="utf-8"))
    for i,ft in enumerate(gj.get("features",[]),1):
        g=shape(ft["geometry"]); p=ft.get("properties",{}) or {}; c=g.centroid
        k=str(p.get("building_type",p.get("kind","review"))).lower()
        if "част" in k or k=="private": k="private"
        elif "мкд" in k or k=="mdu": k="mdu"
        else: k="review"
        out.append(Building(str(p.get("building_id",f"BLD-{i:05d}")),c.x,c.y,k,float(p.get("confidence",1.0)),float(p.get("area_m2",0)),str(p.get("note",""))))
    return out

def buildings_auto(image: Path, georef: Georef, cfg: dict):
    img=cv2.imread(str(image)); maxside=int(cfg.get("work_max_side_px",2200)); sc=min(1,maxside/max(img.shape[:2])); work=cv2.resize(img,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA) if sc<1 else img
    gray=cv2.cvtColor(work,cv2.COLOR_BGR2GRAY); cl=cv2.createCLAHE(2.0,(8,8)).apply(gray); e=cv2.Canny(cl,50,140); e=cv2.morphologyEx(e,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=2)
    cs,_=cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); mpp=georef.mpp()/sc; out=[]; n=0
    amin=float(cfg.get("private_min_area_m2",30)); amax=float(cfg.get("building_max_area_m2",5000)); mdu=float(cfg.get("mdu_min_area_m2",350)); confmin=float(cfg.get("min_active_confidence",0.55))
    for c in cs:
        a=cv2.contourArea(c)*mpp*mpp
        if not (amin<=a<=amax): continue
        rect=cv2.minAreaRect(c); w,h=rect[1];
        if min(w,h)<3: continue
        rectangularity=cv2.contourArea(c)/max(1,w*h); confidence=max(0,min(1,0.35+0.65*rectangularity))
        M=cv2.moments(c);
        if not M["m00"]: continue
        x=(M["m10"]/M["m00"])/sc; y=(M["m01"]/M["m00"])/sc; lon,lat=georef.px2ll(x,y); n+=1
        kind="review" if confidence<confmin else ("mdu" if a>=mdu else "private")
        out.append(Building(f"BLD-{n:05d}",lon,lat,kind,confidence,a,"auto CV baseline"))
    return out

# ---------------- roads / graph ----------------
def roads_geojson(path: Path):
    gj=json.load(open(path,encoding="utf-8")); out=[]
    for ft in gj.get("features",[]):
        g=shape(ft["geometry"])
        if isinstance(g,LineString): out.append(g)
        elif isinstance(g,MultiLineString): out.extend(list(g.geoms))
    return out

def roads_auto(image: Path, georef: Georef, cfg: dict):
    img=cv2.imread(str(image)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); blur=cv2.GaussianBlur(gray,(5,5),0); edges=cv2.Canny(blur,40,110)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=int(cfg.get("hough_threshold",80)),minLineLength=int(cfg.get("min_line_px",40)),maxLineGap=int(cfg.get("max_gap_px",20)))
    out=[]
    if lines is None: return out
    for L in lines[:,0,:]:
        a=georef.px2ll(L[0],L[1]); b=georef.px2ll(L[2],L[3]); out.append(LineString([a,b]))
    return out

def graph_from_roads(roads, snap_m=15):
    lon0=np.mean([p[0] for r in roads for p in r.coords]); lat0=np.mean([p[1] for r in roads for p in r.coords]); mp=Metric(lon0,lat0); G=nx.Graph(); coords={}
    def key(x,y): return f"{round(x,2)}:{round(y,2)}"
    for r in roads:
        pts=[mp.xy(*p) for p in r.coords]
        for a,b in zip(pts[:-1],pts[1:]):
            ka,kb=key(*a),key(*b); coords[ka]=a; coords[kb]=b; G.add_edge(ka,kb,length=math.dist(a,b))
    nodes=list(G.nodes)
    for i,a in enumerate(nodes):
        if G.degree[a]>1: continue
        xa,ya=coords[a]
        best=None
        for b in nodes:
            if a==b: continue
            d=math.dist((xa,ya),coords[b])
            if d<=snap_m and (best is None or d<best[0]): best=(d,b)
        if best and not G.has_edge(a,best[1]): G.add_edge(a,best[1],length=best[0],bridge=True)
    if not nx.is_connected(G):
        c=max(nx.connected_components(G),key=len); G=G.subgraph(c).copy(); coords={k:v for k,v in coords.items() if k in G}
    return G,coords,mp

def nearest_node(G,coords,xy): return min(G.nodes,key=lambda n: math.dist(coords[n],xy))
def path_nodes(G,a,b): return nx.shortest_path(G,a,b,weight="length")
def path_len(G,a,b): return nx.shortest_path_length(G,a,b,weight="length")
def road_median(G,candidates,terminals): return min(candidates,key=lambda c: sum(path_len(G,c,t) for t in terminals))

# ---------------- design ----------------
def split_fat_sizes(n):
    if n<=8: return [n]
    if n<=16: return [n]
    if n<=24: return [16,n-16]
    return [math.ceil(n/2), n-math.ceil(n/2)]

def splitter_for(n): return ("1x8",8) if n<=8 else ("1x16",16)

def design(buildings, roads, cfg):
    active=[b for b in buildings if b.kind in ("private","mdu")]; priv=[b for b in active if b.kind=="private"]; mdus=[b for b in active if b.kind=="mdu"]; review=[b for b in buildings if b.kind=="review"]
    G,coords,mp=graph_from_roads(roads,float(cfg["roads"].get("snap_m",15)))
    bnode={b.id:nearest_node(G,coords,mp.xy(b.lon,b.lat)) for b in active}
    oc=cfg.get("olt",{}); fixed=oc.get("mode","fixed")=="fixed"
    if fixed:
        olon,olat=map(float,oc["point_lonlat"]); olt_node=nearest_node(G,coords,mp.xy(olon,olat)); olt_source="fixed"
    else:
        terms=[bnode[b.id] for b in priv]; sample=list(G.nodes)[::max(1,len(G)//400)] or list(G.nodes); olt_node=road_median(G,sample,terms); olon,olat=mp.ll(*coords[olt_node]); olt_source="auto-review"
    priv_sorted=sorted(priv,key=lambda b:(path_len(G,olt_node,bnode[b.id]),b.id)); maxpon=int(cfg["network"].get("pon_max_private",32)); pon_groups=[priv_sorted[i:i+maxpon] for i in range(0,len(priv_sorted),maxpon)]
    fats=[]; muftas=[]; pons=[]; assigns=[]; features=[]; feeder_edge_load=defaultdict(int)
    def ll(n): return mp.ll(*coords[n])
    for pi,group in enumerate(pon_groups,1):
        pon=f"PON-{pi:03d}"; terminals=[bnode[b.id] for b in group]; center=road_median(G,list(set(terminals)),terminals)
        sizes=split_fat_sizes(len(group)); ordered=sorted(group,key=lambda b:path_len(G,center,bnode[b.id])); branches=[]; k=0
        for sz in sizes: branches.append(ordered[k:k+sz]); k+=sz
        fat_nodes=[]
        for bi,branch in enumerate(branches,1):
            terms=[bnode[b.id] for b in branch]; fn=road_median(G,list(set(terms)),terms); fat_nodes.append(fn); spl,cap=splitter_for(len(branch)); fid=f"FAT-{pi:03d}-{bi}"; lon,lat=ll(fn); fats.append(Fat(fid,pon,spl,cap,len(branch),fn,lon,lat))
            for port,b in enumerate(branch,1):
                bn=bnode[b.id]; npath=path_nodes(G,bn,fn); line=[(b.lon,b.lat)]+[ll(n) for n in npath]; d=math.dist(mp.xy(b.lon,b.lat),coords[bn])+path_len(G,bn,fn); features.append({"type":"Feature","geometry":mapping(LineString(line)),"properties":{"type":"DROP","building_id":b.id,"fat":fid,"pon":pon,"length_m":round(d,1)}}); assigns.append({"building_id":b.id,"pon":pon,"fat":fid,"fat_port":port,"drop_m":d,"lon":b.lon,"lat":b.lat})
        mn=road_median(G,list(set(fat_nodes)),fat_nodes); mlon,mlat=ll(mn); incoming=int(cfg["network"].get("minimum_mufta_incoming_f",8)); mid=f"M-{pi:03d}"; muftas.append(Mufta(mid,pon,mn,mlon,mlat,incoming,incoming,len(fat_nodes)))
        for fat,fn in zip([f for f in fats if f.pon==pon],fat_nodes):
            pn=path_nodes(G,fn,mn); geom=LineString([ll(n) for n in pn]); features.append({"type":"Feature","geometry":mapping(geom),"properties":{"type":"DISTRIBUTION","fat":fat.id,"mufta":mid,"pon":pon,"length_m":round(path_len(G,fn,mn),1)}})
        fp=path_nodes(G,mn,olt_node); fl=path_len(G,mn,olt_node); features.append({"type":"Feature","geometry":mapping(LineString([ll(n) for n in fp])),"properties":{"type":"FEEDER","mufta":mid,"pon":pon,"length_m":round(fl,1)}})
        for a,b in zip(fp[:-1],fp[1:]): feeder_edge_load[tuple(sorted((a,b)))]+=1
        two=len(fat_nodes)>1; losses=cfg["network"]["splitter_loss_db"]; splitloss=float(losses["1x2"] if two else 0)+max(float(losses[f.splitter]) for f in fats if f.pon==pon)
        max_odn=0; max_loss=0
        for a in [x for x in assigns if x["pon"]==pon]:
            fat=next(f for f in fats if f.id==a["fat"]); dist=path_len(G,fat.road_node,mn); odn=a["drop_m"]+dist+fl; loss=splitloss+(odn/1000)*float(cfg["network"].get("fiber_loss_db_km",0.35))+float(cfg["network"].get("connector_loss_db",1))+float(cfg["network"].get("splice_loss_db",0.4))+float(cfg["network"].get("engineering_margin_db",3)); a.update(distribution_m=dist,feeder_m=fl,odn_m=odn,optical_loss_db=loss); max_odn=max(max_odn,odn); max_loss=max(max_loss,loss)
        pons.append({"pon":pon,"private_subscribers":len(group),"fat_ids":";".join(f.id for f in fats if f.pon==pon),"mufta":mid,"split_ratio":32 if two and any(f.capacity==16 for f in fats if f.pon==pon) else max(f.capacity for f in fats if f.pon==pon),"max_odn_m":max_odn,"max_loss_db":max_loss,"status":"OK" if max_loss<=float(cfg["network"].get("optical_budget_db",28)) else "REVIEW"})
    standards=cfg["network"].get("feeder_standard_fibers",[4,8,12,24,48,72,96,144]); reserve=float(cfg["network"].get("feeder_reserve_ratio",0.25)); cable_m=defaultdict(float)
    for a,b,d in G.edges(data=True):
        load=feeder_edge_load.get(tuple(sorted((a,b))),0); cap=std_capacity(max(1,math.ceil(load*(1+reserve))),standards) if load else 0; L=float(d["length"]); cable_m[cap]+=L if cap else 0; features.append({"type":"Feature","geometry":mapping(LineString([ll(a),ll(b)])),"properties":{"type":"ROAD","length_m":round(L,1),"active_pon_fibers":load,"feeder_cable_f":cap}})
    for b in buildings: features.append({"type":"Feature","geometry":mapping(Point(b.lon,b.lat)),"properties":{"type":"BUILDING","building_id":b.id,"building_type":b.kind,"confidence":b.confidence,"area_m2":b.area_m2}})
    for f in fats: features.append({"type":"Feature","geometry":mapping(Point(f.lon,f.lat)),"properties":{"type":"FAT",**asdict(f)}})
    for m in muftas: features.append({"type":"Feature","geometry":mapping(Point(m.lon,m.lat)),"properties":{"type":"MUFTA",**asdict(m)}})
    olon,olat=ll(olt_node); features.append({"type":"Feature","geometry":mapping(Point(olon,olat)),"properties":{"type":"OLT","source":olt_source}})
    mdu_rows=[]
    for i,b in enumerate(mdus,1):
        bn=bnode[b.id]; d=math.dist(mp.xy(b.lon,b.lat),coords[bn]); mdu_rows.append({"building_id":b.id,"pon":f"MDU-PON-{i:03d}","entry_m":d,"status":"dedicated-entry-no-apartment-count"}); features.append({"type":"Feature","geometry":mapping(LineString([(b.lon,b.lat),ll(bn)])),"properties":{"type":"MDU_ENTRY","building_id":b.id,"length_m":round(d,1)}})
    return dict(G=G,coords=coords,metric=mp,private=priv,mdus=mdus,review=review,fats=fats,muftas=muftas,pons=pons,assigns=assigns,mdu_rows=mdu_rows,features=features,cable_m=cable_m,olt=(olon,olat,olt_source))

# ---------------- outputs ----------------
def render(image: Path, georef: Georef, D: dict, out: Path):
    im=Image.open(image).convert("RGBA"); ov=Image.new("RGBA",im.size,(0,0,0,0)); dr=ImageDraw.Draw(ov)
    def px(pt): return georef.ll2px(*pt)
    colors={"DROP":(0,220,255,130),"DISTRIBUTION":(0,110,255,210),"FEEDER":(230,0,255,230)}
    for ft in D["features"]:
        t=ft["properties"].get("type"); g=shape(ft["geometry"])
        if t in colors and isinstance(g,LineString): dr.line([px(p) for p in g.coords],fill=colors[t],width=1 if t=="DROP" else (3 if t=="DISTRIBUTION" else 5))
    for f in D["fats"]:
        x,y=px((f.lon,f.lat)); dr.rectangle((x-4,y-4,x+4,y+4),fill=(255,140,0,255),outline=(0,0,0,255))
    for m in D["muftas"]:
        x,y=px((m.lon,m.lat)); dr.rectangle((x-4,y-4,x+4,y+4),fill=(255,240,0,255),outline=(0,0,0,255))
    for b in D["private"]:
        x,y=px((b.lon,b.lat)); dr.ellipse((x-2,y-2,x+2,y+2),fill=(0,255,100,220))
    x,y=px(D["olt"][:2]); dr.ellipse((x-8,y-8,x+8,y+8),fill=(0,255,0,255),outline=(0,0,0,255),width=2)
    Image.alpha_composite(im,ov).convert("RGB").save(out,quality=92)

def export(cfg, image, georef, D, outdir):
    outdir.mkdir(parents=True,exist_ok=True); dump_json(outdir/"network.geojson",fc(D["features"]))
    dump_csv(outdir/"drops.csv",D["assigns"],["building_id","pon","fat","fat_port","drop_m","distribution_m","feeder_m","odn_m","optical_loss_db","lon","lat"])
    dump_csv(outdir/"fats.csv",[asdict(x) for x in D["fats"]],["id","pon","splitter","capacity","used","road_node","lon","lat"])
    dump_csv(outdir/"muftas.csv",[asdict(x) for x in D["muftas"]],["id","pon","road_node","lon","lat","incoming_f","capacity_splices","fat_branches"])
    dump_csv(outdir/"pons.csv",D["pons"],["pon","private_subscribers","fat_ids","mufta","split_ratio","max_odn_m","max_loss_db","status"])
    cable=[]
    for cap,m in sorted(D["cable_m"].items()): cable.append({"cable_f":cap,"measured_km":m/1000,"purchase_km_8pct":m/1000*1.08})
    dump_csv(outdir/"feeder_cable_by_capacity.csv",cable,["cable_f","measured_km","purchase_km_8pct"])
    qa={"project_name":cfg["project"]["name"],"private":len(D["private"]),"mdu":len(D["mdus"]),"review":len(D["review"]),"all_private_have_one_drop":len(D["assigns"])==len(D["private"]),"fat_over_capacity":sum(f.used>f.capacity for f in D["fats"]),"pon_over_limit":sum(p["private_subscribers"]>int(cfg["network"].get("pon_max_private",32)) for p in D["pons"]),"pon_budget_review":sum(p["status"]!="OK" for p in D["pons"]),"mdu_apartment_counts_invented":False,"olt_source":D["olt"][2],"mufta_rule":"capacity_splices == fully-spliced incoming cable fiber count; one fusion joins two fibers"}
    dump_json(outdir/"qa_report.json",qa)
    render(image,georef,D,outdir/"network_map.png")
    return qa

# ---------------- CLI ----------------
def run(config: Path):
    cfg=load_yaml(config); base=config.parent; tmp=base/".planner_cache"; tmp.mkdir(exist_ok=True)
    src=cfg["project"]["source_image"]; image=local_image(src,tmp); image=(base/image).resolve() if not image.is_absolute() else image; georef=Georef(image,cfg["georef"])
    bc=cfg["buildings"]; buildings=buildings_geojson((base/Path(bc["input_geojson"])).resolve()) if bc.get("mode","geojson")=="geojson" else buildings_auto(image,georef,bc)
    rc=cfg["roads"]; roads=roads_geojson((base/Path(rc["input_geojson"])).resolve()) if rc.get("mode","geojson")=="geojson" else roads_auto(image,georef,rc)
    if not roads: raise RuntimeError("No roads detected; provide roads.geojson or tune auto mode")
    D=design(buildings,roads,cfg); out=(base/Path(cfg["project"].get("output_dir","output"))).resolve(); qa=export(cfg,image,georef,D,out); print(json.dumps(qa,ensure_ascii=False,indent=2)); print("Output:",out)

def init(path: Path):
    sample=Path(__file__).with_name("config.example.yaml")
    if sample.exists(): path.write_text(sample.read_text(encoding="utf-8"),encoding="utf-8")
    else: raise FileNotFoundError("config.example.yaml not found")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); a=sub.add_parser("init"); a.add_argument("path",type=Path); b=sub.add_parser("run"); b.add_argument("config",type=Path); x=ap.parse_args(); init(x.path) if x.cmd=="init" else run(x.config)
