"""Berlin Mitte route search: Flask local server (V1-V44 graph).

Python runs the algorithms (A*, Uniform Cost Search, Greedy Search) and
serves the web page + map. The browser only draws and animates.

Setup (once):   pip install flask
Run:            python graph.py        then open http://127.0.0.1:5000
Keep graph-map.png (or graph-map_1.png) in the same folder as this file.

Endpoints:  /                 the web page
            /map.png          the map image
            /api/graph        nodes and edges as JSON
            /api/search?algo=astar|ucs|greedy&start=V1&goal=V32
"""
import json
import math
import os

from flask import Flask, Response, abort, request, send_file

app = Flask(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
M_PER_UNIT = 1.2                                   # approx metres per map pixel
MAP_CANDIDATES = ("graph-map.png", "graph-map_1.png")

# ---------------------------------------------------------------- graph
NODES = {
    "V1": (140, 458),  "V2": (282, 416),  "V3": (300, 492),  "V4": (373, 483),
    "V5": (366, 404),  "V6": (353, 325),  "V7": (517, 311),  "V8": (522, 387),
    "V9": (616, 298),  "V10": (626, 376), "V11": (650, 611), "V12": (545, 621),
    "V13": (538, 542), "V14": (477, 548), "V15": (469, 473), "V16": (380, 560),
    "V17": (322, 569), "V18": (351, 645), "V19": (387, 642), "V20": (395, 736),
    "V21": (426, 793), "V22": (641, 533), "V23": (633, 457), "V24": (729, 366),
    "V25": (721, 289), "V26": (804, 281), "V27": (815, 357), "V28": (932, 345),
    "V29": (941, 423), "V30": (811, 439), "V31": (993, 330), "V32": (1014, 388),
    "V33": (951, 516), "V34": (916, 568), "V35": (914, 587), "V36": (755, 603),
    "V37": (703, 607), "V38": (760, 668), "V39": (659, 691), "V40": (666, 766),
    "V41": (769, 745), "V42": (923, 664), "V43": (932, 728), "V44": (1111, 628),
}
LETTERS = list(NODES)       # every node can be chosen as start / goal

EDGES = [
    ("V1", "V2"), ("V2", "V5"), ("V5", "V6"), ("V6", "V7"), ("V7", "V9"),
    ("V9", "V25"), ("V25", "V26"), ("V7", "V8"), ("V5", "V8"), ("V8", "V10"),
    ("V9", "V10"), ("V25", "V24"), ("V10", "V24"), ("V24", "V27"),
    ("V27", "V28"), ("V28", "V31"), ("V31", "V32"), ("V28", "V29"),
    ("V29", "V32"), ("V29", "V30"), ("V29", "V33"), ("V33", "V44"),
    ("V33", "V34"), ("V34", "V35"), ("V24", "V36"), ("V10", "V23"),
    ("V23", "V15"), ("V23", "V22"), ("V15", "V4"), ("V15", "V14"),
    ("V4", "V5"), ("V4", "V3"), ("V3", "V2"), ("V3", "V17"), ("V4", "V16"),
    ("V17", "V16"), ("V16", "V14"), ("V14", "V13"), ("V13", "V22"),
    ("V13", "V12"), ("V12", "V11"), ("V22", "V11"), ("V11", "V37"),
    ("V37", "V36"), ("V36", "V35"), ("V35", "V42"), ("V42", "V43"),
    ("V43", "V41"), ("V41", "V40"), ("V40", "V21"), ("V21", "V20"),
    ("V20", "V19"), ("V19", "V18"), ("V18", "V17"), ("V19", "V16"),
    ("V11", "V39"), ("V39", "V40"), ("V36", "V38"), ("V38", "V41"),
]


def dist(a, b):
    return math.hypot(NODES[a][0] - NODES[b][0], NODES[a][1] - NODES[b][1])


ADJ = {k: [] for k in NODES}
for _a, _b in EDGES:
    _w = dist(_a, _b)
    ADJ[_a].append((_b, _w))
    ADJ[_b].append((_a, _w))


def meters(units):
    return round(units * M_PER_UNIT)


# --------------------------------------------------------------- search
ALGOS = {"A* Search": "astar", "Uniform Cost Search": "ucs", "Greedy Search": "greedy"}


def run_search(algo, start, goal):
    """Return (steps, path, length). Each step is (node, parent_or_None)."""
    parent = {start: None}
    cost = {start: 0.0}
    closed = set()
    frontier = [start]
    steps = []

    def key(n):
        if algo == "ucs":
            return cost[n]                      # g(n)
        if algo == "greedy":
            return dist(n, goal)                # h(n)
        return cost[n] + dist(n, goal)          # f(n) = g(n) + h(n)

    while frontier:
        cur = min(frontier, key=key)
        frontier.remove(cur)
        if cur in closed:
            continue
        closed.add(cur)
        steps.append((cur, parent[cur]))
        if cur == goal:
            break
        for nb, w in ADJ[cur]:
            if nb in closed:
                continue
            c = cost[cur] + w
            if nb not in cost or (algo != "greedy" and c < cost[nb]):
                cost[nb] = c
                parent[nb] = cur
            if nb not in frontier:
                frontier.append(nb)

    path, length = [], 0.0
    if goal in closed:
        n = goal
        while n is not None:
            path.insert(0, n)
            n = parent[n]
        length = sum(dist(path[i - 1], path[i]) for i in range(1, len(path)))
    return steps, path, length

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Berlin Mitte route search</title>
<style>
:root{--bg:#fff;--panel:#f3f5fa;--text:#1b1f2a;--line:#d5d9e5;--blue:#2f5bea}
@media(prefers-color-scheme:dark){:root{--bg:#14171f;--panel:#1e2330;--text:#e6e9f2;--line:#323a4e}}
body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}
.tabs{background:var(--blue);display:flex}
.tabs button{background:none;border:0;color:#cfd9ff;font:bold 14px system-ui;padding:11px 18px;cursor:pointer}
.tabs button.on{color:#fff}
.bar{background:var(--panel);padding:8px 12px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
.bar .sp{margin-left:auto}
button.go{background:var(--blue);color:#fff;border:0;border-radius:4px;padding:6px 18px;font-weight:bold;cursor:pointer}
button.go:disabled{opacity:.5}
button.ghost{background:none;color:var(--text);border:1px solid var(--line);border-radius:4px;padding:5px 14px;cursor:pointer}
.wrap{overflow-x:auto}
canvas{display:block;max-width:100%;height:auto;cursor:default}
#info{padding:8px 12px;background:var(--panel);min-height:3.6em}
#info div{margin:2px 0}
table{border-collapse:collapse;width:100%;max-width:560px;margin:8px 0}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
tr.best{background:rgba(40,160,70,.2)}
#cmp{display:none;padding:8px 12px;background:var(--panel)}
</style>
</head>
<body>
<div class="tabs"><button id="t1" class="on">Search algorithm</button><button id="t2">Compare algorithms</button></div>
<div class="bar">
  <label><input type="checkbox" id="showN" checked> Show nodes</label>
  <label><input type="checkbox" id="showE" checked> Show edges</label>
  <label><input type="checkbox" id="showH"> Show heuristics</label>
  <span class="sp"></span>
  <label>Start <select id="start"></select></label>
  <label>Goal <select id="goal"></select></label>
</div>
<div class="wrap"><canvas id="cv" width="1432" height="830"></canvas></div>
<div class="bar" id="searchBar">
  <label>Algorithm <select id="algo"><option value="astar">A* Search</option><option value="ucs">Uniform Cost Search</option><option value="greedy">Greedy Search</option><option value="all">All three (UCS, Greedy, A*) on the map</option></select></label>
  <label>Speed <input type="range" id="speed" min="1" max="10" value="6"></label>
  <span class="sp"></span>
  <button class="ghost" id="clear">Clear</button>
  <button class="go" id="go">Search</button>
</div>
<div id="info"><div id="l1"></div><div id="l2"></div><div id="l3"></div></div>
<div id="cmp">
  <button class="go" id="cmpBtn">Compare all</button>
  <table><thead><tr><th>Algorithm</th><th>Nodes checked</th><th>Route length</th></tr></thead><tbody id="rows"></tbody></table>
  <div id="cmpNote"></div>
</div>
<script>
const MAP_SRC = "/map.png";
const W = 1432, H = 830, M_PER_UNIT = 1.2, DARK = "#1d3a9e";
let NODES = {}, EDGES = [], IDS = [];
const ALGOS = {astar:"A* Search", ucs:"Uniform Cost Search", greedy:"Greedy Search"};
const dist = (a,b) => Math.hypot(NODES[a][0]-NODES[b][0], NODES[a][1]-NODES[b][1]);
const meters = u => Math.round(u*M_PER_UNIT);
async function runSearch(algo, start, goal){
  const r = await fetch("/api/search?algo="+algo+"&start="+start+"&goal="+goal);
  if(!r.ok) throw new Error("server error");
  return r.json();
}
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const img = new Image(); img.onload = draw; img.src = MAP_SRC;

let multi = [], token = 0, running = false, tap = 0, explored = [], routePts = [], routeN = 0, dot = null;

function line(a,b){ ctx.beginPath(); ctx.moveTo(...NODES[a]); ctx.lineTo(...NODES[b]); ctx.stroke(); }
function draw(){
  if(!IDS.length) return;
  ctx.clearRect(0,0,W,H);
  if(img.complete && img.naturalWidth) ctx.drawImage(img,0,0,W,H);
  ctx.lineCap = ctx.lineJoin = "round";
  if($("showE").checked){ ctx.strokeStyle = "#3b4663"; ctx.lineWidth = 3; EDGES.forEach(([a,b]) => line(a,b)); }
  ctx.strokeStyle = DARK; ctx.lineWidth = 5;
  explored.forEach(s => { if(s.parent) line(s.parent, s.node); });
  if(routeN > 1){
    for(const [col,wd] of [["#fff",14],["#ff6a13",8]]){
      ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath(); ctx.moveTo(...routePts[0]);
      for(let i=1;i<routeN;i++) ctx.lineTo(...routePts[i]);
      ctx.stroke();
    }
  }
  multi.forEach(m => { ctx.strokeStyle = m.col; ctx.lineWidth = m.wd; ctx.beginPath(); ctx.moveTo(...NODES[m.path[0]]);
    m.path.forEach(n => ctx.lineTo(...NODES[n])); ctx.stroke(); });
  const seen = new Set(explored.map(s => s.node)), s = $("start").value, g = $("goal").value;
  ctx.font = "bold 10px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for(const k of IDS){
    const isSeen = seen.has(k);
    if(!$("showN").checked && !isSeen) continue;
    const [x,y] = NODES[k];
    ctx.beginPath(); ctx.arc(x,y,14,0,7); ctx.fillStyle = isSeen ? DARK : "#555a66"; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.fillText(k.slice(1), x, y+1);
  }
  if(dot){ ctx.beginPath(); ctx.arc(dot[0],dot[1],10,0,7); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.lineWidth = 5; ctx.strokeStyle = "#ff6a13"; ctx.stroke(); }
  [[s,"#1d3a9e","Start",[]],[g,"#e5322d","Goal",[6,4]]].forEach(([n,col,lab,dash]) => {
    if(lab==="Goal" && g===s) return;
    const [x,y] = NODES[n];
    ctx.setLineDash(dash); ctx.lineWidth = 4; ctx.strokeStyle = col;
    ctx.beginPath(); ctx.arc(x,y,24,0,7); ctx.stroke(); ctx.setLineDash([]);
    ctx.font = "bold 13px system-ui"; ctx.fillStyle = col; ctx.fillText(lab+" ("+n+")", x, y-37);
  });
  if($("showH").checked){
    ctx.font = "bold 10px system-ui"; ctx.fillStyle = "#5a2d82";
    for(const k of IDS) ctx.fillText(meters(dist(k,g))+" m", NODES[k][0], NODES[k][1]+25);
  }
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const delay = () => [900,650,480,360,280,210,150,100,60,30][+$("speed").value-1];
const say = (a="",b="",c="") => { $("l1").textContent = a; $("l2").textContent = b; $("l3").textContent = c; };
const reset = () => say("Click two nodes on the map, or pick them above, then press Search.");

function clearAll(){
  token++; running = false; multi = []; explored = []; routePts = []; routeN = 0; dot = null;
  $("go").disabled = false; reset(); draw();
}

async function search(){
  if(running) return;
  const s = $("start").value, g = $("goal").value;
  if(s === g){ say("Choose a different goal. Start and goal are the same place."); return; }
  clearAll();
  if($("algo").value === "all"){
    const cfg = [["ucs","#2f5bea",18],["greedy","#1f9d3a",12],["astar","#ff6a13",6]], out = [];
    for(const [k,col,wd] of cfg){ const r = await runSearch(k,s,g);
      if(r.path.length){ multi.push({path:r.path,col,wd}); }
      out.push(ALGOS[k]+" ("+col.replace("#2f5bea","blue").replace("#1f9d3a","green").replace("#ff6a13","orange")+"): "+
        (r.path.length ? meters(r.length).toLocaleString()+" m, "+r.steps.length+" nodes checked" : "no route")); }
    say(out[0], out[1], out[2]); draw(); return;
  }
  running = true; $("go").disabled = true;
  const tok = token, name = ALGOS[$("algo").value];
  const {steps, path, length} = await runSearch($("algo").value, s, g);
  say(name+" searching from "+s+" to "+g+"...");
  for(let i=0;i<steps.length;i++){
    explored.push(steps[i]);
    say(name+" searching from "+s+" to "+g+"...", "Nodes checked: "+(i+1)+" (latest: "+steps[i].node+")");
    draw(); await sleep(delay());
    if(tok !== token) return;
  }
  if(!path.length){ say("No route found between "+s+" and "+g+"."); running = false; $("go").disabled = false; return; }
  routePts = [NODES[path[0]]];
  for(let i=1;i<path.length;i++){
    const [x1,y1] = NODES[path[i-1]], [x2,y2] = NODES[path[i]], n = Math.max(1, Math.floor(Math.hypot(x2-x1,y2-y1)/6));
    for(let k=1;k<=n;k++) routePts.push([x1+(x2-x1)*k/n, y1+(y2-y1)*k/n]);
  }
  for(routeN = 1; routeN < routePts.length; routeN += 4){ draw(); await sleep(14); if(tok !== token) return; }
  routeN = routePts.length;
  for(let i=0;i<routePts.length;i+=3){ dot = routePts[i]; draw(); await sleep(14); if(tok !== token) return; }
  dot = routePts[routePts.length-1]; draw();
  say("Route found: "+path.join(" → "), $("l2").textContent, "Distance about "+meters(length).toLocaleString()+" m via "+path.length+" nodes.");
  running = false; $("go").disabled = false;
}

async function compare(){
  const s = $("start").value, g = $("goal").value, body = $("rows");
  body.innerHTML = "";
  if(s === g){ $("cmpNote").textContent = "Choose a different goal first."; return; }
  const rows = await Promise.all(Object.entries(ALGOS).map(async ([k,n]) => { const r = await runSearch(k,s,g); return {n, c:r.steps.length, l:r.length, ok:r.path.length>0}; }));
  const ok = rows.filter(r => r.ok);
  if(!ok.length){ $("cmpNote").textContent = "No route exists from "+s+" to "+g+"."; return; }
  const best = Math.min(...ok.map(r => r.l));
  rows.forEach(r => {
    const tr = body.insertRow(); if(r.ok && Math.abs(r.l-best) < 0.01) tr.className = "best";
    [r.n, r.c, r.ok ? meters(r.l).toLocaleString()+" m" : "-"].forEach(v => tr.insertCell().textContent = v);
  });
  $("cmpNote").textContent = "Highlighted rows found the shortest route from "+s+" to "+g+".";
}

function tab(which){
  $("t1").classList.toggle("on", which===1); $("t2").classList.toggle("on", which===2);
  $("searchBar").style.display = $("info").style.display = which===1 ? "" : "none";
  $("cmp").style.display = which===2 ? "block" : "none";
}

cv.addEventListener("click", e => {
  if(running) return;
  const r = cv.getBoundingClientRect(), x = (e.clientX-r.left)*W/r.width, y = (e.clientY-r.top)*H/r.height;
  const hit = IDS.find(k => Math.hypot(NODES[k][0]-x, NODES[k][1]-y) <= 16);
  if(!hit) return;
  clearAll();
  const other = IDS.find(k => k !== hit);
  if(tap === 0){ $("start").value = hit; if($("goal").value === hit) $("goal").value = other; tap = 1; }
  else { $("goal").value = hit; if($("start").value === hit) $("start").value = other; tap = 0; }
  draw();
});
cv.addEventListener("mousemove", e => {
  const r = cv.getBoundingClientRect(), x = (e.clientX-r.left)*W/r.width, y = (e.clientY-r.top)*H/r.height;
  cv.style.cursor = IDS.some(k => Math.hypot(NODES[k][0]-x, NODES[k][1]-y) <= 16) ? "pointer" : "default";
});
["showN","showE","showH"].forEach(i => $(i).onchange = draw);
["start","goal"].forEach(i => $(i).onchange = clearAll);
$("go").onclick = search; $("clear").onclick = clearAll; $("cmpBtn").onclick = compare;
$("t1").onclick = () => tab(1); $("t2").onclick = () => tab(2);
(async () => {
  try {
    const gr = await (await fetch("/api/graph")).json();
    NODES = gr.nodes; EDGES = gr.edges; IDS = Object.keys(NODES);
    IDS.forEach(k => { $("start").add(new Option(k,k)); $("goal").add(new Option(k,k)); });
    $("start").value = "V1"; $("goal").value = "V32";
    reset(); draw();
  } catch(e){ say("Cannot reach the Python server. Is graph.py still running?"); }
})();
</script>
</body>
</html>
"""


def as_json(obj, code=200):
    # json.dumps keeps the V1, V2, V3... order (jsonify would sort the keys)
    return Response(json.dumps(obj), status=code, mimetype="application/json")


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/map.png")
def map_png():
    for name in MAP_CANDIDATES:
        p = os.path.join(HERE, name)
        if os.path.isfile(p):
            return send_file(p, mimetype="image/png")
    abort(404)


@app.route("/api/graph")
def api_graph():
    return as_json({"nodes": NODES, "edges": EDGES})


@app.route("/api/search")
def api_search():
    algo = request.args.get("algo")
    start = request.args.get("start")
    goal = request.args.get("goal")
    if algo not in ALGOS.values() or start not in NODES or goal not in NODES:
        return as_json({"error": "bad algo, start or goal"}, 400)
    steps, path, length = run_search(algo, start, goal)
    return as_json({"steps": [{"node": n, "parent": p} for n, p in steps],
                    "path": path, "length": length})


if __name__ == "__main__":
    app.run(debug=True)          # http://127.0.0.1:5000