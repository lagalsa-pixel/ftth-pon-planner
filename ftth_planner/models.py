from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class Building:
    id: str
    lon: float
    lat: float
    kind: str  # private | mdu | review
    confidence: float
    area_m2: float = 0.0
    source: str = "auto"
    note: str = ""
    absorbed_households: int = 0

@dataclass
class SubscriberAssignment:
    building_id: str
    pon_id: str
    fat_id: str
    mufta_id: str
    fat_port: int
    splitter: str
    road_node: str
    lateral_m: float
    road_drop_m: float
    drop_m: float
    distribution_m: float
    feeder_m: float
    odn_m: float
    optical_loss_db: float
    margin_db: float
    status: str = "active"

@dataclass
class FatNode:
    fat_id: str
    pon_id: str
    splitter: str
    capacity: int
    used_ports: int
    lon: float
    lat: float
    road_node: str
    p90_drop_m: float = 0.0
    max_drop_m: float = 0.0

@dataclass
class MuftaNode:
    mufta_id: str
    pon_ids: List[str]
    lon: float
    lat: float
    road_node: str
    pon_count: int
    incoming_active_fibers: int
    incoming_cable_f: int
    capacity_splices: int
    stage1_splitters: List[str]
    note: str

@dataclass
class PonTree:
    pon_id: str
    private_subscribers: int
    fat_ids: List[str]
    mufta_id: str
    stage1: str
    split_ratio: int
    branch_capacity: int
    max_odn_m: float
    max_loss_db: float
    min_margin_db: float
    status: str
