PYVIS_COLORS = {
    "function":  "#87CEEB",   # skyblue
    "method":    "#90EE90",   # lightgreen
    "builtin":   "#FA8072",   # salmon
    "library":   "#FFA500",   # orange
    "module":    "#DA70D6",   # orchid
    "variable":  "#FFD700",   # gold
    "callee":    "#FFA500",   # orange 
    "class"   : "#4682B4",    
    "base_class" : "#9370DB",
    "attribute" : "#FFD700",
    "method"   : "#90EE90",   
}

MPL_COLORS = {
    "function":  "skyblue",
    "method":    "lightgreen",
    "builtin":   "salmon",
    "library":   "orange",
    "module":    "orchid",
    "variable":  "gold",
    "callee":    "orange",
    "class"  :   "steelblue",
    "base_class":"mediumpurple",
    "attribute" :"gold",
    "method"  :"lightgreen"
}

BUILTINS = {"open", "print", "len", "getattr", "setattr", "hasattr",
            "isinstance", "type", "range", "enumerate", "zip", "map",
            "filter", "sorted", "list", "dict", "set", "tuple", "str",
            "int", "float", "bool", "repr", "super", "next", "iter",
            "vars", "dir", "id", "hex", "bin", "oct", "abs", "round",
            "min", "max", "sum", "any", "all", "format", "input",
            "exit", "quit", "assert"}

# Shared zoom/toolbar snippet injected into every HTML file
ZOOM_JS = """
<style>
  #zoom-toolbar {
    position: fixed; top: 14px; right: 14px;
    z-index: 9999; display: flex; flex-direction: column; gap: 6px;
  }
  #zoom-toolbar button {
    width: 38px; height: 38px; font-size: 18px; font-weight: bold;
    border: none; border-radius: 6px; background: #2b2b2b; color: #fff;
    cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,.35);
    transition: background .15s;
  }
  #zoom-toolbar button:hover { background: #444; }
  #zoom-toolbar button[title="Fit to screen"] { font-size: 14px; }
</style>
<div id="zoom-toolbar">
  <button title="Zoom in"       onclick="zoomIn()">+</button>
  <button title="Zoom out"      onclick="zoomOut()">&#8722;</button>
  <button title="Fit to screen" onclick="fitScreen()">&#x26F6;</button>
</div>
<script>
var MIN_ZOOM = 0.1, MAX_ZOOM = 3.0, ZOOM_STEP = 0.2;
function getNetwork() {
  if (typeof network !== "undefined") return network;
  for (var k in window)
    if (window[k] && typeof window[k].getScale === "function") return window[k];
  return null;
}
function zoomIn()    { var n=getNetwork(); if(n) n.moveTo({scale:Math.min(n.getScale()+ZOOM_STEP,MAX_ZOOM),animation:{duration:200,easingFunction:"easeInOutQuad"}}); }
function zoomOut()   { var n=getNetwork(); if(n) n.moveTo({scale:Math.max(n.getScale()-ZOOM_STEP,MIN_ZOOM),animation:{duration:200,easingFunction:"easeInOutQuad"}}); }
function fitScreen() { var n=getNetwork(); if(n) n.fit({animation:{duration:500,easingFunction:"easeInOutQuad"}}); }
(function wait(){ var n=getNetwork(); if(!n){setTimeout(wait,200);return;} n.once("stabilized",fitScreen); })();
</script>
"""

PYVIS_OPTIONS = """
{
  "interaction": {
    "zoomView": true,
    "navigationButtons": true,
    "keyboard": { "enabled": true }
  },
  "physics": {
    "enabled": false,
    "barnesHut": {
      "gravitationalConstant": -4000,
      "centralGravity": 0.2,
      "springLength": 180,
      "springConstant": 0.05,
      "damping": 0.09
    },
    "stabilization": { "iterations": 200 }
  },
  "edges": {
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
    "smooth": { "type": "dynamic" }
  }
}
"""

# Extra colors for class diagram node types



# ── Start / end node highlight colors ─────────────────────────────────────────
START_COLOR_MPL   = "limegreen"
END_COLOR_MPL     = "tomato"
START_COLOR_PYVIS = "#32CD32"   # limegreen hex
END_COLOR_PYVIS   = "#FF6347"   # tomato hex