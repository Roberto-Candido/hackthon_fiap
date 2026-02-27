from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json

import matplotlib.pyplot as plt
from ultralytics import YOLO


# -----------------------------
# CONFIG (MUDA SÓ ISSO)
# -----------------------------
IMAGE_NAME = "000057.png"
IMAGE_PATH = Path("imagens_teste") / IMAGE_NAME

RUNS_BASE = Path("runs")   # onde o Ultralytics salva treino: runs/detect/...
OUT_DIR = Path("runs_graph")

IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.7

# "Stride arquitetural" = camadas por quantização do eixo X
N_LAYERS = 4

# Regras simples de conexão (ajuste depois)
ALLOWED = {
    "user": ["api", "server"],
    "external": ["api", "server"],
    "api": ["server", "queue", "database", "storage"],
    "server": ["server", "queue", "database", "storage"],
    "queue": ["server"],
    "database": [],
    "storage": [],
}

# Limite de saídas por tipo (pra não virar “tudo com tudo”)
OUT_DEGREE = {
    "user": 1,
    "external": 1,
    "api": 2,
    "server": 2,
    "queue": 1,
    "database": 0,
    "storage": 0,
}


# -----------------------------
# DATA STRUCTURES
# -----------------------------
@dataclass
class Det:
    idx: int
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return (self.x2 - self.x1)

    @property
    def h(self) -> float:
        return (self.y2 - self.y1)


# -----------------------------
# WEIGHTS
# -----------------------------
def find_best_weights_most_recent(base_runs: Path) -> Path:
    """
    Pega o best.pt mais recente dentro de runs/
    Resolve automaticamente yolo_components, yolo_components2, etc.
    """
    candidates = list(base_runs.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("❌ Não achei nenhum best.pt dentro de runs/ (treino não salvou pesos?)")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# -----------------------------
# DETECTION
# -----------------------------
def run_detection(image_path: Path) -> Tuple[List[Det], Dict[int, str], Path, Tuple[int, int]]:
    """
    Roda YOLO na imagem e retorna:
    - dets: lista de detecções com bbox/conf/classe
    - names: dict id->nome da classe (do modelo)
    - saved_pred_dir: pasta onde Ultralytics salvou a imagem anotada
    - (img_w, img_h): tamanho da imagem inferida (aprox, via r.orig_shape)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")

    weights = find_best_weights_most_recent(RUNS_BASE)
    print(f"✅ Pesos usados: {weights}")

    model = YOLO(str(weights))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        save=True,
        project=str(OUT_DIR),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]
    names = r.names  # dict class_id -> name
    # orig_shape é (h,w)
    img_h, img_w = r.orig_shape[0], r.orig_shape[1]

    dets: List[Det] = []
    if r.boxes is not None and len(r.boxes) > 0:
        for i, b in enumerate(r.boxes):
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append(Det(
                idx=i,
                cls=names.get(cls_id, str(cls_id)),
                conf=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

    saved_pred_dir = OUT_DIR / "pred"
    return dets, names, saved_pred_dir, (img_w, img_h)


# -----------------------------
# GRAPH BUILDING
# -----------------------------
def assign_layers(dets: List[Det], img_w: int, n_layers: int) -> Dict[int, int]:
    layer_w = img_w / float(n_layers)
    layers: Dict[int, int] = {}
    for d in dets:
        layer = int(d.cx // layer_w)
        layer = max(0, min(n_layers - 1, layer))
        layers[d.idx] = layer
    return layers


def build_edges(dets: List[Det]) -> List[Tuple[int, int]]:
    """
    Cria arestas inferidas:
    - só conecta para a direita (cx maior)
    - respeita ALLOWED por tipo
    - pega até OUT_DEGREE conexões por nó
    - escolhe alvos mais próximos no eixo X
    """
    edges: List[Tuple[int, int]] = []
    dets_sorted = sorted(dets, key=lambda d: d.cx)

    for src in dets_sorted:
        allowed_targets = ALLOWED.get(src.cls, [])
        k = OUT_DEGREE.get(src.cls, 1)

        if k <= 0 or not allowed_targets:
            continue

        candidates = [
            dst for dst in dets_sorted
            if dst.cx > src.cx and dst.cls in allowed_targets
        ]
        candidates.sort(key=lambda d: (d.cx - src.cx))

        for dst in candidates[:k]:
            edges.append((src.idx, dst.idx))

    # remove duplicatas mantendo ordem
    edges = list(dict.fromkeys(edges))
    return edges


def to_dot(dets: List[Det], edges: List[Tuple[int, int]], layers: Dict[int, int]) -> str:
    id_to = {d.idx: d for d in dets}
    lines = ["digraph G {", "rankdir=LR;"]
    # nós
    for d in dets:
        l = layers.get(d.idx, 0)
        lines.append(f'n{d.idx} [label="{d.cls}\\n{d.conf:.2f}\\nL{l}"];')
    # arestas
    for a, b in edges:
        lines.append(f"n{a} -> n{b};")
    lines.append("}")
    return "\n".join(lines)


# -----------------------------
# GRAPH DRAWING (PNG)
# -----------------------------
def draw_graph_png(
    dets: List[Det],
    edges: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    out_path: Path,
):
    """
    Desenha o grafo usando as posições reais da imagem (cx, cy).
    O grafo vai bater visualmente com o diagrama detectado.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)  # inverte Y para bater com imagem
    ax.set_axis_off()

    # Desenha arestas
    for a, b in edges:
        src = dets[a]
        dst = dets[b]
        ax.annotate(
            "",
            xy=(dst.cx, dst.cy),
            xytext=(src.cx, src.cy),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )

    # Desenha nós
    for d in dets:
        label = f"{d.cls}\n{d.conf:.2f}\nL{layers[d.idx]}"
        ax.text(
            d.cx,
            d.cy,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", linewidth=2),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print(f"🖼️ Imagem: {IMAGE_PATH.resolve()}")

    dets, names, pred_dir, (img_w, img_h) = run_detection(IMAGE_PATH)

    print("\n📦 Detecções:")
    if not dets:
        print("  (nenhuma detecção)")
        raise SystemExit(0)

    for d in sorted(dets, key=lambda x: x.cx):
        print(f"  - {d.cls:10s} | conf={d.conf:.3f} | xyxy=({d.x1:.1f},{d.y1:.1f},{d.x2:.1f},{d.y2:.1f})")

    layers = assign_layers(dets, img_w=img_w, n_layers=N_LAYERS)
    edges = build_edges(dets)

    # salva JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_json = [dict(**asdict(d), cx=d.cx, cy=d.cy, layer=layers.get(d.idx, 0)) for d in dets]
    edges_json = [{"from": a, "to": b} for a, b in edges]

    (OUT_DIR / "nodes.json").write_text(json.dumps(nodes_json, indent=2), encoding="utf-8")
    (OUT_DIR / "edges.json").write_text(json.dumps(edges_json, indent=2), encoding="utf-8")

    # salva DOT
    dot = to_dot(dets, edges, layers)
    (OUT_DIR / "graph.dot").write_text(dot, encoding="utf-8")

    # DESENHO DO GRAFO (PNG)
    graph_png = OUT_DIR / "graph.png"
    draw_graph_png(
        dets,
        edges,
        img_w=img_w,
        img_h=img_h,
        out_path=graph_png,
    )

    print("\n🔗 Arestas inferidas:")
    for a, b in edges:
        print(f"  {a} -> {b}   ({dets[a].cls} -> {dets[b].cls})")

    print("\n✅ Saídas geradas:")
    print(f"  - Imagem anotada (YOLO): {pred_dir.resolve()}")
    print(f"  - Nodes JSON:            {(OUT_DIR / 'nodes.json').resolve()}")
    print(f"  - Edges JSON:            {(OUT_DIR / 'edges.json').resolve()}")
    print(f"  - DOT:                   {(OUT_DIR / 'graph.dot').resolve()}")
    print(f"  - PNG do grafo:          {graph_png.resolve()}")
