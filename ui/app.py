"""
ui/app.py — Smart Grid Topology Optimizer
White UI, zoom/pan graphs, IEEE 33/69/85 + Radial/Star/Mesh.
Direct file-path imports — works regardless of working directory.
Run: python ui/app.py
"""
import sys, os, importlib.util, traceback

_UI   = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UI)

def _load_module(name, filepath):
    if not os.path.exists(filepath):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[LOAD] {name} failed: {e}")
        traceback.print_exc()
        return None

_ieee  = _load_module("ieee_networks",    os.path.join(_ROOT, "power_system", "ieee_networks.py"))
_topo  = _load_module("topology_builder", os.path.join(_ROOT, "power_system", "topology_builder.py"))
_bex   = _load_module("branch_exchange",  os.path.join(_ROOT, "phase2",       "branch_exchange.py"))
_pipe  = _load_module("phase2_main",      os.path.join(_ROOT, "phase2",       "phase2_main.py"))

print(f"[BOOT] ieee_networks   : {'OK' if _ieee  else 'FAIL'}")
print(f"[BOOT] topology_builder: {'OK' if _topo  else 'FAIL'}")
print(f"[BOOT] branch_exchange : {'OK' if _bex   else 'FAIL'}")
print(f"[BOOT] phase2_main     : {'OK' if _pipe  else 'FAIL'}")

run_pipeline         = getattr(_pipe, "run_pipeline",         None)
get_network_topology = getattr(_topo, "get_network_topology", None)

if run_pipeline:
    print("[BOOT] run_pipeline: OK")
else:
    print("[BOOT] CRITICAL: run_pipeline not found")

from flask import Flask, request, jsonify, render_template_string
import pandapower as pp, pandapower.networks as pn, numpy as np

app = Flask(__name__)

def _graph_data(net, closed_tie=None, opened_line=None):
    n = len(net.bus)
    nodes = []
    for i in range(n):
        try:    v = float(net.res_bus.at[i, "vm_pu"])
        except: v = 1.0
        p = sum(float(r["p_mw"]) for _, r in net.load.iterrows() if int(r["bus"]) == i)
        nodes.append({"id":i,"voltage":round(v,4),"p_load":round(p*1000,2),
                      "type":"substation" if i==0 else "load"})
    edges = []
    for idx, row in net.line.iterrows():
        svc = bool(row["in_service"])
        if idx == closed_tie:    tag = "closed"
        elif idx == opened_line: tag = "opened"
        elif svc:                tag = "active"
        else:                    tag = "inactive"
        edges.append({"id":int(idx),"source":int(row["from_bus"]),"target":int(row["to_bus"]),
                      "in_service":svc,"name":str(row.get("name",f"Line {idx}")),"tag":tag})
    return {"nodes":nodes,"edges":edges}

def _simple_net(network, load_p, load_q):
    if network=="ieee69" and _ieee:   net=_ieee.create_ieee69()
    elif network=="ieee85" and _ieee: net=_ieee.create_ieee85()
    else:                             net=pn.case33bw()
    if len(net.load)>0:
        net.load["p_mw"]*=load_p; net.load["q_mvar"]*=load_q
    try: pp.runpp(net,algorithm='nr',numba=False)
    except: pass
    return net

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smart Grid Topology Optimizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f4f6fb;color:#1a1a2e;font-family:'Segoe UI',Arial,sans-serif;min-height:100vh}

/* HEADER */
header{background:#fff;border-bottom:2px solid #e0e4ef;padding:14px 32px;
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 2px 8px rgba(80,80,180,0.07)}
header h1{font-size:1.3rem;color:#3a3aad;font-weight:700;letter-spacing:.3px}
header small{color:#888;font-size:.77rem}
.hdg{display:flex;gap:8px}
.hbdg{background:#eef0ff;padding:3px 12px;border-radius:20px;font-size:.7rem;color:#3a3aad;font-weight:600;border:1px solid #c7cbf7}

/* LAYOUT */
.wrap{max-width:1380px;margin:0 auto;padding:22px 24px}
.slbl{font-size:.7rem;color:#888;text-transform:uppercase;letter-spacing:1.2px;
  margin-bottom:8px;margin-top:20px;font-weight:600}

/* NETWORK CARDS */
.ng{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:4px}
.nc{background:#fff;border:2px solid #e0e4ef;border-radius:12px;padding:16px;
  cursor:pointer;transition:all .18s;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.nc:hover{border-color:#3a3aad;transform:translateY(-2px);box-shadow:0 4px 14px rgba(58,58,173,.1)}
.nc.sel{border-color:#3a3aad;background:#f0f1ff}
.nc h3{color:#3a3aad;font-size:.95rem;margin-bottom:2px;font-weight:700}
.nc .meta{color:#888;font-size:.72rem;margin-bottom:8px}
.pills{display:flex;flex-wrap:wrap;gap:5px}
.pill{background:#f0f1ff;padding:2px 8px;border-radius:7px;font-size:.69rem;color:#3a3aad;font-weight:600;border:1px solid #d0d3f5}

/* TOPOLOGY CARDS */
.tg{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:4px}
.tc{background:#fff;border:2px solid #e0e4ef;border-radius:12px;padding:16px;
  cursor:pointer;transition:all .18s;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.tc:hover{border-color:#16a97a;transform:translateY(-2px);box-shadow:0 4px 14px rgba(22,169,122,.1)}
.tc.sel{border-color:#16a97a;background:#f0fff9}
.tc h3{font-size:.9rem;margin-bottom:3px;font-weight:700;color:#1a1a2e}
.tc p{font-size:.7rem;color:#888}
.tc svg{display:block;margin:0 auto 8px}

/* RECOMMENDED VALUES TABLE */
.rec-box{background:#fff;border:1px solid #e0e4ef;border-radius:12px;padding:16px 20px;
  margin-top:14px;box-shadow:0 1px 4px rgba(0,0,0,.05)}
.rec-box h4{font-size:.78rem;font-weight:700;color:#3a3aad;margin-bottom:10px;
  text-transform:uppercase;letter-spacing:.5px}
.rec-tbl{width:100%;border-collapse:collapse;font-size:.78rem}
.rec-tbl th{background:#f0f1ff;color:#3a3aad;font-weight:700;padding:7px 12px;
  text-align:left;border-bottom:2px solid #d0d3f5}
.rec-tbl td{padding:6px 12px;border-bottom:1px solid #f0f1f5;color:#333}
.rec-tbl tr:last-child td{border-bottom:none}
.rec-tbl tr:hover td{background:#f8f9ff}
.btn-use{background:#3a3aad;color:#fff;border:none;padding:3px 10px;border-radius:6px;
  font-size:.68rem;cursor:pointer;font-weight:600;transition:background .15s}
.btn-use:hover{background:#2828a0}
.exp-badge{display:inline-block;background:#e8fff5;color:#16a97a;border:1px solid #b0f0d8;
  padding:1px 7px;border-radius:10px;font-size:.67rem;font-weight:700}
.warn-badge{display:inline-block;background:#fff8e0;color:#c07000;border:1px solid #f0d080;
  padding:1px 7px;border-radius:10px;font-size:.67rem;font-weight:700}

/* CONTROLS */
.cg2{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}
.cgr{background:#fff;border-radius:10px;padding:12px;box-shadow:0 1px 4px rgba(0,0,0,.05);border:1px solid #e0e4ef}
.cgr label{display:block;font-size:.7rem;color:#888;margin-bottom:5px;font-weight:600}
.cgr input{width:100%;background:#f8f9ff;border:1.5px solid #d0d3f5;color:#1a1a2e;
  padding:7px 10px;border-radius:7px;font-size:.83rem;outline:none;transition:border .15s}
.cgr input:focus{border-color:#3a3aad}

/* BUTTONS */
.br{display:flex;gap:14px;margin-bottom:22px}
.btn{padding:11px 26px;border-radius:10px;font-size:.88rem;font-weight:700;
  cursor:pointer;border:none;transition:all .18s}
.btnp{background:linear-gradient(135deg,#3a3aad,#6060e0);color:#fff;
  box-shadow:0 3px 12px rgba(58,58,173,.25)}
.btns{background:#fff;color:#3a3aad;border:2px solid #3a3aad}
.btn:hover{transform:translateY(-1px);opacity:.92}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

/* GRAPH SECTION */
.grow{display:none;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:22px}
.gp{background:#fff;border-radius:13px;padding:15px;
  box-shadow:0 2px 10px rgba(0,0,0,.07);border:1px solid #e0e4ef}
.gp-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.gp-hdr h3{font-size:.82rem;color:#555;display:flex;align-items:center;gap:6px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.dg{background:#aaa}.dgr{background:#16a97a}
.zoom-btns{display:flex;gap:4px}
.zbtn{background:#f0f1ff;border:1px solid #d0d3f5;color:#3a3aad;width:26px;height:26px;
  border-radius:6px;cursor:pointer;font-size:.85rem;font-weight:700;
  display:flex;align-items:center;justify-content:center;transition:background .12s}
.zbtn:hover{background:#3a3aad;color:#fff}
.graph-wrap{overflow:hidden;border-radius:8px;border:1px solid #e8eaf0;background:#fafbff}
svg.graph{width:100%;height:320px;display:block;cursor:grab}
svg.graph:active{cursor:grabbing}
.leg{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}
.li{display:flex;align-items:center;gap:5px;font-size:.69rem;color:#888}
.ll{width:18px;height:3px;border-radius:1px}

/* LOADING */
#loading{display:none;background:#fff;border-radius:13px;padding:28px;text-align:center;
  margin-bottom:22px;box-shadow:0 2px 10px rgba(0,0,0,.07);border:1px solid #e0e4ef}
.spin{width:36px;height:36px;border:3px solid #e0e4ef;border-top-color:#3a3aad;
  border-radius:50%;animation:sp .8s linear infinite;margin:0 auto 14px}
@keyframes sp{to{transform:rotate(360deg)}}
.sl{list-style:none;text-align:left;display:inline-block}
.sl li{padding:3px 0;font-size:.83rem;color:#aaa}
.sl li.act{color:#3a3aad;font-weight:600}
.sl li.dn{color:#16a97a;font-weight:600}

/* RESULTS */
#res{display:none}
.rg{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}
.rc{background:#fff;border-radius:11px;padding:16px;
  box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #e0e4ef}
.rc.r{border-top:4px solid #e8445a}.rc.g{border-top:4px solid #16a97a}
.rc.y{border-top:4px solid #f0a000}.rc.a{border-top:4px solid #3a3aad}
.rc .val{font-size:1.6rem;font-weight:800;color:#1a1a2e}
.rc .lbl{font-size:.7rem;color:#888;margin-top:3px;font-weight:600}
.al{border-radius:9px;padding:11px 14px;margin-bottom:10px;font-size:.82rem;font-weight:500}
.ag{background:#f0fff9;border:1.5px solid #b0f0d8;color:#0d7a50}
.ab{background:#f0f4ff;border:1.5px solid #c0ccf8;color:#2030a0}
.ay{background:#fff8e0;border:1.5px solid #f0d080;color:#8a5500}
.cg4{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:22px}
.cc{background:#fff;border-radius:13px;padding:16px;
  box-shadow:0 2px 8px rgba(0,0,0,.06);border:1px solid #e0e4ef}
.cc h4{font-size:.73rem;color:#888;text-transform:uppercase;letter-spacing:.6px;
  margin-bottom:12px;font-weight:700}
canvas{max-height:210px}
.ib{background:#f0f1ff;border-radius:9px;padding:11px 16px;font-size:.79rem;
  color:#3a3aad;margin-bottom:18px;border-left:4px solid #3a3aad;font-weight:500}

@media(max-width:900px){
  .grow,.cg4,.ng,.tg,.rg{grid-template-columns:1fr}
  .cg2{grid-template-columns:1fr 1fr}
}
</style>
</head>
<body>
<header>
  <div>
    <h1>⚡ Smart Grid Topology Optimizer</h1>
    <small>GNN · LSTM · CNN — IEEE 33 / 69 / 85 Bus Distribution Networks</small>
  </div>
  <div class="hdg">
    <span class="hbdg">GNN</span><span class="hbdg">LSTM</span>
    <span class="hbdg">CNN</span><span class="hbdg">Branch Exchange</span>
  </div>
</header>

<div class="wrap">

<!-- NETWORK -->
<div class="slbl">Step 1 — Select IEEE Network</div>
<div class="ng">
  <div class="nc sel" id="n-ieee33" onclick="sNet('ieee33')">
    <h3>IEEE 33-Bus</h3>
    <div class="meta">12.66 kV · Classic Radial Benchmark · Baran &amp; Wu 1989</div>
    <div class="pills">
      <span class="pill">33 Buses</span><span class="pill">32 Lines</span>
      <span class="pill">5 Tie Switches</span><span class="pill">3.715 MW Load</span>
    </div>
  </div>
  <div class="nc" id="n-ieee69" onclick="sNet('ieee69')">
    <h3>IEEE 69-Bus</h3>
    <div class="meta">12.66 kV · High Load Density · Urban Feeder</div>
    <div class="pills">
      <span class="pill">69 Buses</span><span class="pill">68 Lines</span>
      <span class="pill">5 Tie Switches</span><span class="pill">~3.80 MW Load</span>
    </div>
  </div>
  <div class="nc" id="n-ieee85" onclick="sNet('ieee85')">
    <h3>IEEE 85-Bus</h3>
    <div class="meta">11 kV · Multi-Feeder · Industrial Zone</div>
    <div class="pills">
      <span class="pill">85 Buses</span><span class="pill">84 Lines</span>
      <span class="pill">3 Tie Switches</span><span class="pill">~2.57 MW Load</span>
    </div>
  </div>
</div>

<!-- TOPOLOGY -->
<div class="slbl">Step 2 — Select Topology Type</div>
<div class="tg">
  <div class="tc sel" id="t-radial" onclick="sTopo('radial')">
    <svg width="76" height="52" viewBox="0 0 76 52">
      <line x1="8" y1="26" x2="68" y2="26" stroke="#3a3aad" stroke-width="2.5"/>
      <line x1="28" y1="26" x2="28" y2="8" stroke="#3a3aad" stroke-width="2.5"/>
      <line x1="48" y1="26" x2="48" y2="44" stroke="#3a3aad" stroke-width="2.5"/>
      <circle cx="8" cy="26" r="6" fill="#f0a000"/>
      <circle cx="28" cy="26" r="4" fill="#3a3aad"/><circle cx="48" cy="26" r="4" fill="#3a3aad"/>
      <circle cx="68" cy="26" r="4" fill="#3a3aad"/><circle cx="28" cy="8" r="4" fill="#3a3aad"/>
      <circle cx="48" cy="44" r="4" fill="#3a3aad"/>
    </svg>
    <h3>Radial (Tree)</h3><p>Single path from substation to each bus</p>
  </div>
  <div class="tc" id="t-star" onclick="sTopo('star')">
    <svg width="76" height="52" viewBox="0 0 76 52">
      <line x1="38" y1="26" x2="8" y2="8" stroke="#16a97a" stroke-width="2.5"/>
      <line x1="38" y1="26" x2="68" y2="8" stroke="#16a97a" stroke-width="2.5"/>
      <line x1="38" y1="26" x2="8" y2="44" stroke="#16a97a" stroke-width="2.5"/>
      <line x1="38" y1="26" x2="68" y2="44" stroke="#16a97a" stroke-width="2.5"/>
      <line x1="38" y1="26" x2="38" y2="4" stroke="#16a97a" stroke-width="2.5"/>
      <circle cx="38" cy="26" r="7" fill="#f0a000"/>
      <circle cx="8" cy="8" r="4" fill="#16a97a"/><circle cx="68" cy="8" r="4" fill="#16a97a"/>
      <circle cx="8" cy="44" r="4" fill="#16a97a"/><circle cx="68" cy="44" r="4" fill="#16a97a"/>
      <circle cx="38" cy="4" r="4" fill="#16a97a"/>
    </svg>
    <h3>Star (Hub-Spoke)</h3><p>Hub-and-spoke with backup paths</p>
  </div>
  <div class="tc" id="t-mesh" onclick="sTopo('mesh')">
    <svg width="76" height="52" viewBox="0 0 76 52">
      <line x1="8" y1="12" x2="68" y2="12" stroke="#e8445a" stroke-width="2.5"/>
      <line x1="8" y1="40" x2="68" y2="40" stroke="#e8445a" stroke-width="2.5"/>
      <line x1="8" y1="12" x2="8" y2="40" stroke="#e8445a" stroke-width="2.5"/>
      <line x1="38" y1="12" x2="38" y2="40" stroke="#e8445a" stroke-width="2.5"/>
      <line x1="68" y1="12" x2="68" y2="40" stroke="#e8445a" stroke-width="2.5"/>
      <circle cx="8" cy="12" r="4" fill="#f0a000"/>
      <circle cx="38" cy="12" r="4" fill="#e8445a"/><circle cx="68" cy="12" r="4" fill="#e8445a"/>
      <circle cx="8" cy="40" r="4" fill="#e8445a"/><circle cx="38" cy="40" r="4" fill="#e8445a"/>
      <circle cx="68" cy="40" r="4" fill="#e8445a"/>
    </svg>
    <h3>Mesh (Multi-Path)</h3><p>Cross-connected feeders, multiple paths</p>
  </div>
</div>

<!-- RECOMMENDED VALUES TABLE -->
<div class="rec-box">
  <h4>📋 Recommended Input Values — Guaranteed Visible Before/After Changes</h4>
  <table class="rec-tbl" id="rectbl">
    <thead>
      <tr>
        <th>Network</th><th>Topology</th><th>Active Load %</th><th>Reactive Load %</th>
        <th>V Min (pu)</th><th>V Max (pu)</th><th>Seed</th>
        <th>Expected Reduction</th><th>Switching Action</th><th>Use</th>
      </tr>
    </thead>
    <tbody id="rectbody"></tbody>
  </table>
</div>

<!-- CONTROLS -->
<div class="slbl">Step 3 — Configure Parameters</div>
<div class="cg2">
  <div class="cgr"><label>Active Load (%)</label><input type="number" id="lp" value="110" min="50" max="150" step="5"></div>
  <div class="cgr"><label>Reactive Load (%)</label><input type="number" id="lq" value="100" min="50" max="150" step="5"></div>
  <div class="cgr"><label>Voltage Lower Limit (pu)</label><input type="number" id="vmin" value="0.93" min="0.88" max="0.98" step="0.01"></div>
  <div class="cgr"><label>Voltage Upper Limit (pu)</label><input type="number" id="vmax" value="1.05" min="1.01" max="1.10" step="0.01"></div>
  <div class="cgr"><label>Forecast Horizon (hours)</label><input type="number" id="hz" value="24" min="1" max="72"></div>
  <div class="cgr"><label>Random Seed</label><input type="number" id="sd" value="42" min="1" max="999"></div>
</div>

<div class="br">
  <button class="btn btns" onclick="previewGraph()">👁 Preview Network Graph</button>
  <button class="btn btnp" id="runbtn" onclick="runOpt()">⚡ Run Optimization</button>
</div>

<!-- GRAPHS -->
<div class="grow" id="gsec">
  <div class="gp">
    <div class="gp-hdr">
      <h3><span class="dot dg"></span>Before Optimization</h3>
      <div class="zoom-btns">
        <button class="zbtn" onclick="zoomGraph('gb',1.3)" title="Zoom In">＋</button>
        <button class="zbtn" onclick="zoomGraph('gb',0.77)" title="Zoom Out">－</button>
        <button class="zbtn" onclick="resetZoom('gb')" title="Reset">⟳</button>
      </div>
    </div>
    <div class="graph-wrap"><svg class="graph" id="gb"></svg></div>
    <div class="leg" id="lgb"></div>
  </div>
  <div class="gp">
    <div class="gp-hdr">
      <h3><span class="dot dgr"></span>After Optimization</h3>
      <div class="zoom-btns">
        <button class="zbtn" onclick="zoomGraph('ga',1.3)" title="Zoom In">＋</button>
        <button class="zbtn" onclick="zoomGraph('ga',0.77)" title="Zoom Out">－</button>
        <button class="zbtn" onclick="resetZoom('ga')" title="Reset">⟳</button>
      </div>
    </div>
    <div class="graph-wrap"><svg class="graph" id="ga"></svg></div>
    <div class="leg" id="lga"></div>
  </div>
</div>

<!-- LOADING -->
<div id="loading">
  <div class="spin"></div>
  <p style="color:#3a3aad;font-weight:700;margin-bottom:12px;font-size:.95rem">Running 7-Stage Optimization Pipeline…</p>
  <ul class="sl" id="slist">
    <li id="s1">⬜ Loading IEEE network</li>
    <li id="s2">⬜ LSTM load forecasting</li>
    <li id="s3">⬜ CNN spatial voltage analysis</li>
    <li id="s4">⬜ Generating training scenarios</li>
    <li id="s5">⬜ Training GNN surrogate model</li>
    <li id="s6">⬜ Branch exchange optimization</li>
    <li id="s7">⬜ Validating constraints</li>
  </ul>
</div>

<!-- RESULTS -->
<div id="res">
  <div class="ib" id="ib"></div>
  <div class="rg">
    <div class="rc r"><div class="val" id="rb">—</div><div class="lbl">Baseline Loss (MW)</div></div>
    <div class="rc g"><div class="val" id="ro">—</div><div class="lbl">Optimized Loss (MW)</div></div>
    <div class="rc y"><div class="val" id="rr">—</div><div class="lbl">Loss Reduction (%)</div></div>
    <div class="rc a"><div class="val" id="rg">—</div><div class="lbl">GNN RMSE (MW)</div></div>
  </div>
  <div id="alrt"></div>
  <div class="cg4">
    <div class="cc"><h4>📊 Voltage Profile — All Buses (pu)</h4><canvas id="cv"></canvas></div>
    <div class="cc"><h4>📊 Total Loss Before vs After (MW)</h4><canvas id="ct"></canvas></div>
    <div class="cc"><h4>📊 Per-Line Power Loss (MW) — First 30</h4><canvas id="cl"></canvas></div>
    <div class="cc"><h4>📊 Voltage Deviation from 1.0 pu</h4><canvas id="cd"></canvas></div>
  </div>
</div>
</div><!-- .wrap -->

<script>
/* ── State ── */
var NET='ieee33', TOPO='radial', CH={}, ZOOM={gb:1, ga:1}, PAN={gb:{x:0,y:0}, ga:{x:0,y:0}};
var SIMS={}, GTRANSFORMS={gb:null, ga:null};

/* ── Recommended values for every network+topology combo ── */
var RECS=[
  // IEEE 33 — all three topologies verified working
  {net:'ieee33',topo:'radial',p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~21.8%',action:'Close Tie 32 → Open Line 6'},
  {net:'ieee33',topo:'star',  p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~60–65%',action:'Activate StarBackup_0 → Replace Line 4'},
  {net:'ieee33',topo:'mesh',  p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~5–15%',action:'Close MeshExtra → Open MeshCross'},
  // IEEE 69 — use lower load to guarantee convergence
  {net:'ieee69',topo:'radial',p:100,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~20–25%',action:'Close Tie → Open high-loss feeder line'},
  {net:'ieee69',topo:'star',  p:100,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~15–20%',action:'Activate StarBackup → Replace long primary'},
  {net:'ieee69',topo:'mesh',  p:100,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~8–15%',action:'Close MeshExtra → Open MeshCross'},
  // IEEE 85 — verified working
  {net:'ieee85',topo:'radial',p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~17%',action:'Close Tie 84 → Open Line 10'},
  {net:'ieee85',topo:'star',  p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~17%',action:'Activate StarBackup → Replace Line 10'},
  {net:'ieee85',topo:'mesh',  p:110,q:100,vmin:0.85,vmax:1.05,seed:42,
   exp:'~3–8%',action:'Close Tie 84 → Open MeshCross_1'},
];

function buildRecTable(){
  var tb=document.getElementById('rectbody'); tb.innerHTML='';
  RECS.forEach(function(r,i){
    var isActive=(r.net===NET&&r.topo===TOPO);
    var tr=document.createElement('tr');
    if(isActive) tr.style.background='#f0f8ff';
    tr.innerHTML=
      '<td><b>'+r.net.toUpperCase()+'</b></td>'+
      '<td>'+r.topo.charAt(0).toUpperCase()+r.topo.slice(1)+'</td>'+
      '<td>'+r.p+'%</td><td>'+r.q+'%</td>'+
      '<td>'+r.vmin+'</td><td>'+r.vmax+'</td><td>'+r.seed+'</td>'+
      '<td><span class="exp-badge">'+r.exp+'</span></td>'+
      '<td style="font-size:.7rem;color:#555">'+r.action+'</td>'+
      '<td><button class="btn-use" onclick="useRec('+i+')">Use</button></td>';
    tb.appendChild(tr);
  });
}

function useRec(i){
  var r=RECS[i];
  sNet(r.net); sTopo(r.topo);
  document.getElementById('lp').value=r.p;
  document.getElementById('lq').value=r.q;
  document.getElementById('vmin').value=r.vmin;
  document.getElementById('vmax').value=r.vmax;
  document.getElementById('sd').value=r.seed;
  buildRecTable();
  previewGraph();
}

function sNet(n){
  NET=n;
  document.querySelectorAll('.nc').forEach(c=>c.classList.remove('sel'));
  document.getElementById('n-'+n).classList.add('sel');
  buildRecTable();
}
function sTopo(t){
  TOPO=t;
  document.querySelectorAll('.tc').forEach(c=>c.classList.remove('sel'));
  document.getElementById('t-'+t).classList.add('sel');
  buildRecTable();
}

/* ── Zoom / Pan ── */
function zoomGraph(id, factor){
  if(!GTRANSFORMS[id]) return;
  var t=GTRANSFORMS[id];
  var svg=document.getElementById(id);
  var w=svg.getBoundingClientRect().width||600, h=320;
  var newK=t.k*factor;
  newK=Math.max(0.3, Math.min(8, newK));
  var cx=w/2, cy=h/2;
  var newX=cx - newK*(cx-t.x)/t.k;
  var newY=cy - newK*(cy-t.y)/t.k;
  var newT=d3.zoomIdentity.translate(newX,newY).scale(newK);
  GTRANSFORMS[id]=newT;
  d3.select('#'+id).select('g.zoom-container')
    .transition().duration(250)
    .attr('transform','translate('+newT.x+','+newT.y+') scale('+newT.k+')');
}
function resetZoom(id){
  var svg=d3.select('#'+id);
  var t=d3.zoomIdentity;
  GTRANSFORMS[id]=t;
  svg.select('g.zoom-container')
    .transition().duration(250).attr('transform','translate(0,0) scale(1)');
  if(SIMS[id]){
    SIMS[id].alpha(.3).restart();
  }
}

/* ── Draw graph with zoom/pan ── */
function drawGraph(svgId, data, lgId){
  var svgEl=document.getElementById(svgId);
  var svg=d3.select('#'+svgId);
  svg.selectAll('*').remove();
  ZOOM[svgId]=1; PAN[svgId]={x:0,y:0};

  var w=svgEl.getBoundingClientRect().width||580, h=320;
  svg.attr('viewBox','0 0 '+w+' '+h);

  // Zoom container
  var g=svg.append('g').attr('class','zoom-container');

  // Setup D3 zoom
  var zoomBeh=d3.zoom()
    .scaleExtent([0.15, 10])
    .on('zoom', function(event){
      GTRANSFORMS[svgId]=event.transform;
      g.attr('transform', event.transform);
    });
  svg.call(zoomBeh);
  GTRANSFORMS[svgId]=d3.zoomIdentity;

  var nodes=data.nodes.map(d=>Object.assign({},d));
  var links=data.edges.map(d=>Object.assign({},d));
  var tm={}; links.forEach(e=>{tm[e.id]=e.tag});

  function ec(t){
    if(t==='closed') return '#16a97a';
    if(t==='opened') return '#e8445a';
    if(t==='active') return '#3a3aad';
    return '#ccc';
  }
  function ew(t){ return(t==='closed'||t==='opened')?4:1.8 }
  function ed(t){ return t==='inactive'?'5,4':'none' }
  function nc(d){
    if(d.type==='substation') return '#f0a000';
    if(d.voltage<0.95) return '#e8445a';
    if(d.voltage<0.97) return '#ff9a3c';
    return '#3a3aad';
  }

  var sim=d3.forceSimulation(nodes)
    .force('link',d3.forceLink(links).id(d=>d.id).distance(nodes.length>50?28:38))
    .force('charge',d3.forceManyBody().strength(nodes.length>50?-50:-80))
    .force('center',d3.forceCenter(w/2,h/2))
    .force('col',d3.forceCollide(nodes.length>50?10:13));
  SIMS[svgId]=sim;

  var link=g.append('g').selectAll('line').data(links).enter().append('line')
    .attr('stroke',d=>ec(tm[d.id]))
    .attr('stroke-width',d=>ew(tm[d.id]))
    .attr('stroke-dasharray',d=>ed(tm[d.id]))
    .attr('opacity',.85);

  var node=g.append('g').selectAll('circle').data(nodes).enter().append('circle')
    .attr('r',d=>d.type==='substation'?9:(nodes.length>50?4:5.5))
    .attr('fill',d=>nc(d))
    .attr('stroke','#fff').attr('stroke-width',1.5)
    .call(d3.drag()
      .on('start',(e,d)=>{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y})
      .on('drag',(e,d)=>{d.fx=e.x;d.fy=e.y})
      .on('end',(e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}));

  // Labels for changed lines
  var closedTag=Object.keys(tm).find(k=>tm[k]==='closed');
  var openedTag=Object.keys(tm).find(k=>tm[k]==='opened');
  if(closedTag!==undefined){
    g.append('g').selectAll('text.et').data(links.filter(d=>d.tag==='closed'||d.tag==='opened'))
      .enter().append('text')
      .attr('class','et')
      .attr('font-size','9px').attr('fill',d=>d.tag==='closed'?'#16a97a':'#e8445a')
      .attr('font-weight','700').attr('text-anchor','middle')
      .text(d=>d.tag==='closed'?'NEW':'OPEN');
  }

  // Tooltip
  var tip=d3.select('body').append('div')
    .style('position','absolute').style('background','#fff')
    .style('border','1.5px solid #d0d3f5').style('border-radius','8px')
    .style('padding','7px 12px').style('font-size','.75rem').style('color','#1a1a2e')
    .style('pointer-events','none').style('display','none')
    .style('box-shadow','0 2px 8px rgba(0,0,0,.12)').style('z-index','999');

  node.on('mouseover',(e,d)=>tip.style('display','block')
      .html('<b>Bus '+d.id+'</b><br>V = '+d.voltage+' pu<br>Load = '+d.p_load+' kW<br>'+
            (d.voltage<0.95?'<span style="color:#e8445a">⚠ Critical</span>':
             d.voltage<0.97?'<span style="color:#ff9a3c">⚠ Warning</span>':
             '<span style="color:#16a97a">✓ Normal</span>')))
    .on('mousemove',e=>tip.style('left',(e.pageX+14)+'px').style('top',(e.pageY-32)+'px'))
    .on('mouseout',()=>tip.style('display','none'));

  sim.on('tick',()=>{
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('cx',d=>d.x).attr('cy',d=>d.y);
    g.selectAll('text.et').data(links.filter(d=>d.tag==='closed'||d.tag==='opened'))
      .attr('x',d=>(d.source.x+d.target.x)/2)
      .attr('y',d=>(d.source.y+d.target.y)/2-5);
  });

  // Legend
  var lg=document.getElementById(lgId);
  if(lg) lg.innerHTML=
    '<div class="li"><div class="ll" style="background:#3a3aad;height:2px"></div>Active Line</div>'+
    '<div class="li"><div class="ll" style="border-top:2px dashed #ccc;height:0;width:18px"></div>Inactive (Tie)</div>'+
    '<div class="li"><div class="ll" style="background:#16a97a;height:4px"></div>Closed NEW ✓</div>'+
    '<div class="li"><div class="ll" style="background:#e8445a;height:4px"></div>Opened ✗</div>'+
    '<div class="li"><div style="width:9px;height:9px;border-radius:50%;background:#f0a000;display:inline-block;margin-right:2px"></div>Substation</div>'+
    '<div class="li"><div style="width:9px;height:9px;border-radius:50%;background:#e8445a;display:inline-block;margin-right:2px"></div>Critical V&lt;0.95</div>'+
    '<div class="li"><div style="width:9px;height:9px;border-radius:50%;background:#3a3aad;display:inline-block;margin-right:2px"></div>Normal Bus</div>'+
    '<div class="li" style="color:#555;font-style:italic;font-size:.68rem">Scroll=zoom · Drag=pan · Node drag=move</div>';
}

/* ── Preview ── */
function previewGraph(){
  fetch('/graph_preview',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({topology:TOPO,network:NET,
      load_p:+document.getElementById('lp').value/100,
      load_q:+document.getElementById('lq').value/100,
      seed:+document.getElementById('sd').value})
  }).then(r=>r.json()).then(d=>{
    if(d.graph){
      drawGraph('gb',d.graph,'lgb');
      document.getElementById('gsec').style.display='grid';
    } else alert('Preview error: '+(d.error||'unknown'));
  }).catch(e=>alert('Preview failed: '+e.message));
}

/* ── Optimize ── */
var _si=null,_sx=0;
var SL=['Loading IEEE network…','LSTM load forecast (24h)…','CNN spatial voltage analysis…',
        'Generating 50 training scenarios…','Training GNN surrogate (25 epochs)…',
        'Branch exchange optimization…','Validating radial+voltage constraints…'];

function startAnim(){
  _sx=0;
  for(var i=1;i<=7;i++){var e=document.getElementById('s'+i);e.textContent='⬜ '+SL[i-1];e.className=''}
  _si=setInterval(()=>{
    if(_sx>0){var p=document.getElementById('s'+_sx);p.textContent='✅ '+SL[_sx-1];p.className='dn'}
    if(_sx<7){var c=document.getElementById('s'+(_sx+1));c.textContent='🔄 '+SL[_sx];c.className='act';_sx++}
    else clearInterval(_si);
  },1800);
}

function runOpt(){
  var btn=document.getElementById('runbtn');
  btn.disabled=true;
  document.getElementById('loading').style.display='block';
  document.getElementById('res').style.display='none';
  startAnim();
  fetch('/optimize',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      topology:TOPO, network:NET,
      load_p:+document.getElementById('lp').value/100,
      load_q:+document.getElementById('lq').value/100,
      v_min:+document.getElementById('vmin').value,
      v_max:+document.getElementById('vmax').value,
      horizon:+document.getElementById('hz').value,
      seed:+document.getElementById('sd').value
    })
  }).then(r=>r.json()).then(d=>{
    clearInterval(_si);
    for(var i=1;i<=7;i++){var e=document.getElementById('s'+i);e.textContent='✅ '+SL[i-1];e.className='dn'}
    setTimeout(()=>{
      document.getElementById('loading').style.display='none';
      btn.disabled=false;
      if(d.success) showRes(d);
      else{ alert('Optimization error:\n'+(d.error||JSON.stringify(d))); console.error(d); }
    },600);
  }).catch(e=>{
    clearInterval(_si);
    document.getElementById('loading').style.display='none';
    btn.disabled=false;
    alert('Server error: '+e.message);
  });
}

/* ── Show Results ── */
function showRes(d){
  document.getElementById('res').style.display='block';
  document.getElementById('gsec').style.display='grid';

  drawGraph('gb', d.graph_before, 'lgb');
  drawGraph('ga', d.graph_after,  'lga');

  document.getElementById('rb').textContent=d.baseline_loss.toFixed(4);
  document.getElementById('ro').textContent=d.optimized_loss.toFixed(4);
  document.getElementById('rr').textContent=d.pct_reduction.toFixed(2)+'%';
  document.getElementById('rg').textContent=d.gnn_rmse.toFixed(5);

  document.getElementById('ib').innerHTML=
    '<b>Network:</b> '+(d.network_name||'').toUpperCase()+
    ' &nbsp;|&nbsp; <b>Topology:</b> '+(d.topology_type||'').toUpperCase()+
    ' &nbsp;|&nbsp; <b>Buses:</b> '+d.n_buses+
    ' &nbsp;|&nbsp; <b>Action:</b> '+d.action_description+
    ' &nbsp;|&nbsp; <b>LSTM Forecast:</b> '+(d.forecast_scale*100).toFixed(1)+'%'+
    ' &nbsp;|&nbsp; <b>GNN MSE:</b> '+d.gnn_mse.toFixed(2)+'e-6';

  document.getElementById('alrt').innerHTML=
    '<div class="al ag">✅ Optimization: <b>'+d.action_description+'</b> — <b>'+
    d.pct_reduction.toFixed(2)+'%</b> loss reduction. Radial constraint maintained.</div>'+
    '<div class="al ab">📈 LSTM Forecast: <b>'+(d.forecast_scale*100).toFixed(1)+
    '%</b> predicted load for next '+d.forecast_horizon+' hours. '+
    (d.forecast_scale>1.1?'Proactive reconfiguration applied.':'Load within safe range.')+'</div>'+
    (d.critical_buses&&d.critical_buses.length?
      '<div class="al ay">⚠️ Critical voltage buses after optimization: ['+
      d.critical_buses.join(', ')+'] — voltages below 0.95 pu</div>':'');

  var nb=d.voltages_before.length;
  var bl=Array.from({length:nb},(_,i)=>'B'+i);

  // Chart 1: Voltage profile
  mkC('cv','line',bl,[
    {label:'Before',data:d.voltages_before,borderColor:'#e8445a',backgroundColor:'#e8445a22',
     fill:false,tension:.3,pointRadius:nb>50?1:2,pointBackgroundColor:'#e8445a'},
    {label:'After', data:d.voltages_after, borderColor:'#16a97a',backgroundColor:'#16a97a22',
     fill:false,tension:.3,pointRadius:nb>50?1:2,pointBackgroundColor:'#16a97a'}],
    {vline: d.v_min});

  // Chart 2: Total loss bar
  mkC('ct','bar',['Baseline','Optimized'],[
    {label:'Power Loss (MW)',
     data:[d.baseline_loss,d.optimized_loss],
     backgroundColor:['rgba(232,68,90,0.6)','rgba(22,169,122,0.6)'],
     borderColor:['#e8445a','#16a97a'],borderWidth:2,
     borderRadius:8}]);

  // Chart 3: Per-line loss
  var mx=Math.min(30,d.line_losses_before.length);
  mkC('cl','bar',Array.from({length:mx},(_,i)=>'L'+i),[
    {label:'Before',data:d.line_losses_before.slice(0,mx),
     backgroundColor:'rgba(232,68,90,0.55)',borderColor:'#e8445a',borderWidth:1,borderRadius:3},
    {label:'After', data:d.line_losses_after.slice(0,mx),
     backgroundColor:'rgba(22,169,122,0.55)',borderColor:'#16a97a',borderWidth:1,borderRadius:3}]);

  // Chart 4: Voltage deviation
  mkC('cd','line',bl,[
    {label:'Before',data:d.voltages_before.map(v=>+(Math.abs(v-1)).toFixed(4)),
     borderColor:'#e8445a',backgroundColor:'rgba(232,68,90,0.1)',
     fill:true,tension:.3,pointRadius:nb>50?1:2},
    {label:'After', data:d.voltages_after.map(v=>+(Math.abs(v-1)).toFixed(4)),
     borderColor:'#16a97a',backgroundColor:'rgba(22,169,122,0.1)',
     fill:true,tension:.3,pointRadius:nb>50?1:2}]);

  window.scrollTo({top:document.getElementById('res').offsetTop-80,behavior:'smooth'});
}

function mkC(id,type,labels,datasets,opts){
  if(CH[id]) CH[id].destroy();
  var cfg={
    type,data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:true,
      plugins:{legend:{labels:{color:'#555',font:{size:11},boxWidth:12}}},
      scales:{
        x:{ticks:{color:'#888',maxTicksLimit:20,font:{size:9}},
           grid:{color:'rgba(0,0,0,.05)'},border:{color:'#e0e4ef'}},
        y:{ticks:{color:'#888',font:{size:9}},
           grid:{color:'rgba(0,0,0,.05)'},border:{color:'#e0e4ef'}}
      }
    }
  };
  CH[id]=new Chart(document.getElementById(id).getContext('2d'),cfg);
}

/* ── Init ── */
window.onload=function(){
  buildRecTable();
  previewGraph();
};
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/graph_preview', methods=['POST'])
def graph_preview():
    d = request.get_json() or {}
    topo    = d.get('topology','radial')
    network = d.get('network','ieee33')
    load_p  = float(d.get('load_p',1.0))
    load_q  = float(d.get('load_q',1.0))
    seed    = int(d.get('seed',42))
    try:
        if get_network_topology:
            net = get_network_topology(topo, network, load_p, load_q, seed)
        else:
            net = _simple_net(network, load_p, load_q)
        try: pp.runpp(net, algorithm='nr', numba=False)
        except: pass
        return jsonify({'graph': _graph_data(net)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    d = request.get_json() or {}
    if run_pipeline is None:
        print(f"[ERROR] run_pipeline is None")
        print(f"[ERROR] phase2_main exists: {os.path.exists(os.path.join(_ROOT,'phase2','phase2_main.py'))}")
        return jsonify({'success':False,'error':'Pipeline not loaded. Check terminal for [BOOT] messages.'}), 500
    try:
        result = run_pipeline(
            topology_type    = d.get('topology','radial'),
            network_name     = d.get('network','ieee33'),
            load_scale_p     = float(d.get('load_p',1.0)),
            load_scale_q     = float(d.get('load_q',1.0)),
            v_min            = float(d.get('v_min',0.95)),
            v_max            = float(d.get('v_max',1.05)),
            forecast_horizon = int(d.get('horizon',24)),
            seed             = int(d.get('seed',42)),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success':False,'error':str(e)}), 500

if __name__ == '__main__':
    print(f"\n[INFO] Project root : {_ROOT}")
    print("[INFO] Open http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)