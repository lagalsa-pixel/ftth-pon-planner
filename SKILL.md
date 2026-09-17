# FTTH/PON planning workflow skill

Use this workflow when the user provides a satellite/orthophoto image for a new settlement and asks for an FTTH/PON design.

1. Never draw the final network before subscriber and road layers are quantified.
2. Prefer GeoTIFF or explicit georeferencing. If only a screenshot exists, request or derive control points/bbox and mark accuracy limits.
3. Build three building classes: private, MDU, review. Do not silently convert uncertain roofs into subscribers.
4. Treat authoritative household counts as QA constraints, not license to fabricate structures.
5. Prefer road vectors or a specialized road segmentation layer. Repair only small graph gaps within configured snap distance and list them in QA.
6. OLT location must be user-confirmed or field/GIS-confirmed. Auto choice is a candidate only.
7. Hard FTTH/PON constraints: one confirmed private house -> one drop -> one FAT port; FAT 1x8/1x16; private PON <=32; two FAT branches use PLC 1x2; MDU uses dedicated building entry until apartment count is known.
8. Route drop/distribution/feeder on the road graph; store geometry first, render second.
9. Optical budget is checked per PON worst case.
10. Cable sizing uses active fibers + reserve and standard cable sizes.
11. Closure capacity rule for this workflow: capacity in fusion splices equals the number of fibers in the fully-spliced incoming cable; one fusion joins two fibers. Example: 24F -> 24 splices.
12. Final map must be deterministic GIS rendering from exported network geometry, never generative artwork.
13. Specification must include individual drop lengths, FAT occupancy, PON loading, mufta incoming cable/capacity, cable lengths by fiber count, and QA exceptions.
14. Before final delivery verify: one drop per private subscriber, no FAT over capacity, no PON over split limit, optical budget exceptions explained, MDU apartment counts not invented, and all totals reconcile.
