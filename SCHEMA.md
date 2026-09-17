# Схема входных GeoJSON

## buildings.geojson

Геометрия может быть `Point` или `Polygon`. Для `Polygon` используется centroid.

Рекомендуемые свойства:

```json
{
  "building_id": "BLD-00001",
  "building_type": "private",
  "confidence": 0.93,
  "area_m2": 112.4,
  "note": ""
}
```

`building_type`: `private`, `mdu`, `review`. Также распознаются русские подписи `частный дом`, `МКД`.

## roads.geojson

`LineString`/`MultiLineString` в WGS84 (EPSG:4326). Имена улиц не обязательны для трассировки, но могут храниться в properties.

## OLT

В YAML:

```yaml
olt:
  mode: fixed
  point_lonlat: [82.12345, 50.12345]
```

Для `mode: auto` создается только расчетный кандидат; его надо подтвердить.
