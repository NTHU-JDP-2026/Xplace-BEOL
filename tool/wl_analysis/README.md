# Wire-length analysis of two routed placements

Why does one placement route longer than another? Run these on the routed DEFs of two runs of a design
(e.g. Innovus vs Xplace), in the Xplace container (`wl_decomp.py` needs FLUTE from `cpp_to_py`):

1. `net_cats.py a.def b.def` — routed length by net category. If much of the difference is in nets
   that exist in one DEF only or have different pins, post-placement optimization rebuilt the netlist
   differently; compare runs with the same netlist first (Innovus flow: `CTS_ONLY=1`).
2. `wl_decomp.py cells.lef a.def b.def [--big]` — per net degree: HPWL, RSMT (Steiner) and routed
   length. RSMT / HPWL is the placement's pin topology, routed / RSMT the router's detour.
3. `sink_spread.py a.def b.def net ...` — how closely the flops and the sinks of given nets sit.

`--scale 4` (default) for DEFs of 4x LEFs (NS3K Innovus). `defio.py` holds the shared DEF/LEF reading.
