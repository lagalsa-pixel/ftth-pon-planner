from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from .models import Building
from .base import MetricProjector

def build_road_graph(roads_lonlat: Sequence[LineString], snap_m: float = 15.0):
    """Build a noded metric road graph. Only disconnected endpoints may be bridged within snap_m; every synthetic bridge is exposed to QA."""
    if not roads_lonlat:
        raise ValueError("No road lines")
    all_pts = [c for ln in roads_lonlat for c in ln.coords]
    lon0 = float(np.mean([p[0] for p in all_pts])); lat0 = float(np.mean([p[1] for p in all_pts]))
    proj = MetricProjector(lon0, lat0)
    metric_lines = [proj.geom_to_metric(ln) for ln in roads_lonlat]
    noded = unary_union(metric_lines)
    segs=[]
    if isinstance(noded, LineString): segs=[noded]
    elif isinstance(noded, MultiLineString): segs=list(noded.geoms)
    else:
        for g in getattr(noded, "geoms", []):
            if isinstance(g, LineString): segs.append(g)
    G=nx.Graph(); node_coords: Dict[str, Tuple[float,float]]={}; key_to_id={}; edge_geoms={}
    def get_node(x,y):
        q=(round(float(x),1), round(float(y),1))
        if q not in key_to_id:
            nid=f"R{len(key_to_id)+1:06d}"; key_to_id[q]=nid; node_coords[nid]=(float(x),float(y)); G.add_node(nid,x=float(x),y=float(y))
        return key_to_id[q]
    for ln in segs:
        coords=list(ln.coords)
        if len(coords)<2: continue
        for a,b in zip(coords[:-1],coords[1:]):
            na=get_node(*a); nb=get_node(*b)
            if na==nb: continue
            d=math.hypot(b[0]-a[0], b[1]-a[1])
            if d<0.05: continue
            e=tuple(sorted((na,nb)))
            if (not G.has_edge(na,nb)) or d < float(G[na][nb].get("weight",1e99)):
                G.add_edge(na,nb,weight=d,length_m=d,bridge=False)
                edge_geoms[e]=LineString([node_coords[na],node_coords[nb]])
    bridge_edges=[]
    if snap_m and snap_m>0 and G.number_of_nodes()>1:
        while True:
            comps=list(nx.connected_components(G))
            if len(comps)<=1: break
            comp_id={n:i for i,c in enumerate(comps) for n in c}
            endpoints=[n for n in G.nodes if G.degree[n] <= 1]
            best=None
            for i,a in enumerate(endpoints):
                xa,ya=node_coords[a]
                for b in endpoints[i+1:]:
                    if comp_id[a]==comp_id[b]: continue
                    xb,yb=node_coords[b]; d=math.hypot(xb-xa,yb-ya)
                    if d<=snap_m and (best is None or d<best[0]): best=(d,a,b)
            if best is None: break
            d,a,b=best; e=tuple(sorted((a,b)))
            G.add_edge(a,b,weight=d,length_m=d,bridge=True)
            edge_geoms[e]=LineString([node_coords[a],node_coords[b]])
            bridge_edges.append(e)
    return G, node_coords, edge_geoms, proj, bridge_edges

def nearest_road_node(x: float, y: float, node_coords: Dict[str, Tuple[float,float]]) -> Tuple[str,float]:
    best=None; bd=1e99
    for nid,(nx_,ny_) in node_coords.items():
        d=(nx_-x)**2+(ny_-y)**2
        if d<bd: bd=d; best=nid
    return best, math.sqrt(bd)

def graph_path_geom(G:nx.Graph, node_coords:dict, src:str, dst:str) -> Tuple[LineString,float,List[Tuple[str,str]]]:
    if src == dst:
        x,y=node_coords[src]
        return LineString([(x,y),(x+0.01,y+0.01)]),0.0,[]
    path=nx.shortest_path(G,src,dst,weight="weight")
    coords=[node_coords[n] for n in path]; length=0.0; edges=[]
    for a,b in zip(path[:-1],path[1:]):
        length += float(G[a][b]["length_m"]); edges.append(tuple(sorted((a,b))))
    return LineString(coords),length,edges

def choose_olt(buildings: Sequence[Building], G, node_coords, proj:MetricProjector, cfg:dict) -> Tuple[str,float,float,str]:
    mode=cfg.get("mode","fixed")
    if mode=="fixed":
        pt=cfg.get("point_lonlat")
        if not pt or len(pt)!=2: raise ValueError("olt.mode=fixed requires olt.point_lonlat=[lon,lat]")
        x,y=proj.xy(float(pt[0]),float(pt[1])); node,_=nearest_road_node(x,y,node_coords); nx_,ny_=node_coords[node]; lon,lat=proj.ll(nx_,ny_)
        return node,lon,lat,"fixed_user_point"
    active=[b for b in buildings if b.kind in ("private","mdu")]
    if not active: raise ValueError("No active buildings for OLT selection")
    ax=[]; ay=[]
    for b in active:
        x,y=proj.xy(b.lon,b.lat); ax.append(x); ay.append(y)
    cx=float(np.median(ax)); cy=float(np.median(ay))
    if mode=="candidates":
        cands=cfg.get("candidates",[])
        if not cands: raise ValueError("olt.mode=candidates requires resolved candidate points")
        scored=[]
        for i,c in enumerate(cands):
            lon=float(c["lon"]); lat=float(c["lat"]); x,y=proj.xy(lon,lat); node,snap=nearest_road_node(x,y,node_coords)
            priority=float(c.get("priority",0.0)); confirmed=bool(c.get("confirmed",False)); score=(-priority, math.hypot(x-cx,y-cy), snap, i)
            scored.append((score,node,confirmed,c.get("id",f"candidate-{i+1}")))
        scored.sort(key=lambda z:z[0]); _,node,confirmed,cid=scored[0]; nx_,ny_=node_coords[node]; lon,lat=proj.ll(nx_,ny_)
        return node,lon,lat,f"candidate:{cid}:"+("confirmed" if confirmed else "NEEDS_FIELD_VALIDATION")
    if mode!="auto": raise ValueError(f"Unknown olt.mode={mode}")
    node,_=nearest_road_node(cx,cy,node_coords); nx_,ny_=node_coords[node]; lon,lat=proj.ll(nx_,ny_)
    return node,lon,lat,"auto_network_median_NEEDS_FIELD_VALIDATION"
