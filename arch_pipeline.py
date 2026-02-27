# arch_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import json
import re
import os
import time
import logging

from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import easyocr


# ======================================================
# LOGGING
# ======================================================
def _setup_logger() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("arch_pipeline")
    if not logger.handlers:
        h = logging.StreamHandler()
        f = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        h.setFormatter(f)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


log = _setup_logger()


# ======================================================
# CONFIG DEFAULTS
# ======================================================
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.7
N_LAYERS = 4

ALLOWED = {
    "user": ["api", "server"],
    "external": ["api", "server"],
    "api": ["server", "queue", "database", "storage"],
    "server": ["server", "queue", "database", "storage"],
    "queue": ["server"],
    "database": [],
    "storage": [],
}

OUT_DEGREE = {
    "user": 1,
    "external": 1,
    "api": 2,
    "server": 2,
    "queue": 1,
    "database": 0,
    "storage": 0,
}


# ======================================================
# DATA STRUCTURES
# ======================================================
@dataclass
class Det:
    idx: int
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    name: str = ""  # OCR label (opcional)

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


# ======================================================
# OCR (EasyOCR)
# ======================================================
_OCR_READER: easyocr.Reader | None = None


def _get_ocr_reader() -> easyocr.Reader:
    """
    Lazy init (uma vez por processo).
    """
    global _OCR_READER
    if _OCR_READER is None:
        langs = os.getenv("OCR_LANGS", "en,pt").split(",")
        langs = [l.strip() for l in langs if l.strip()]
        gpu = os.getenv("OCR_GPU", "0") == "1"
        log.info("OCR: inicializando EasyOCR Reader | langs=%s gpu=%s", langs, gpu)
        _OCR_READER = easyocr.Reader(langs, gpu=gpu)
    return _OCR_READER


def _clean_text(txt: str) -> str:
    txt = (txt or "").strip()
    txt = re.sub(r"\s+", " ", txt)
    # remove lixo mas mantém coisas úteis
    txt = re.sub(r"[^\w\s\-\.\(\)\/\:]", "", txt)
    txt = txt.strip()
    return txt[:120]


def _preprocess(img: Image.Image) -> Image.Image:
    """
    Pré-processamento simples (barato) pra OCR de label.
    """
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    # Upscale pra melhorar OCR em fontes pequenas
    g = g.resize((g.size[0] * 2, g.size[1] * 2))
    return g


def _crop(base: Image.Image, x1: int, y1: int, x2: int, y2: int) -> Image.Image | None:
    W, H = base.size
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return base.crop((x1, y1, x2, y2))


def _ocr_text(img: Image.Image) -> str:
    """
    ✅ EasyOCR NÃO aceita PIL.Image direto.
    ✅ Converte para numpy array.
    """
    reader = _get_ocr_reader()

    arr = np.array(img)  # <- CORREÇÃO DO SEU ERRO
    texts = reader.readtext(arr, detail=0, paragraph=False)

    if not texts:
        return ""
    best = max(texts, key=lambda s: len(s or ""))
    return _clean_text(best)


def _ocr_label_for_det(base_img: Image.Image, det: Det, debug_dir: Path | None = None) -> str:
    """
    MVP:
    - tenta ler label ABAIXO do ícone
    - se vazio, tenta À DIREITA do ícone
    - pega o melhor (mais longo)
    """
    # Região abaixo
    pad_x = int(max(6, det.w * 0.20))
    y_gap = int(max(2, det.h * 0.05))
    region_h = int(max(18, det.h * 0.60))

    below = _crop(
        base_img,
        int(det.x1 - pad_x),
        int(det.y2 + y_gap),
        int(det.x2 + pad_x),
        int(det.y2 + y_gap + region_h),
    )

    txt_below = ""
    if below is not None:
        p = _preprocess(below)
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            p.save(debug_dir / f"det{det.idx:02d}_below.png")
        txt_below = _ocr_text(p)

    # Região direita
    x_gap = int(max(2, det.w * 0.08))
    region_w = int(max(60, det.w * 1.60))
    pad_y = int(max(6, det.h * 0.15))

    right = _crop(
        base_img,
        int(det.x2 + x_gap),
        int(det.y1 - pad_y),
        int(det.x2 + x_gap + region_w),
        int(det.y2 + pad_y),
    )

    txt_right = ""
    if right is not None:
        p = _preprocess(right)
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            p.save(debug_dir / f"det{det.idx:02d}_right.png")
        txt_right = _ocr_text(p)

    # Decide
    candidates = [t for t in [txt_below, txt_right] if t and len(t) >= 2]
    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def enrich_dets_with_ocr_labels(image_path: Path, dets: List[Det]) -> List[Det]:
    """
    Preenche det.name via OCR.

    Ative crops debug:
      SAVE_OCR_DEBUG=1  -> salva em runs_ocr_debug/

    Logs:
      - INFO: OK/FAIL por det + resumo
      - DEBUG: tempos, bbox etc
    """
    t0 = time.time()

    if not dets:
        log.info("OCR: dets=0 (pula)")
        return dets

    base_img = Image.open(image_path).convert("RGB")

    save_debug = os.getenv("SAVE_OCR_DEBUG", "0") == "1"
    debug_dir = Path("runs_ocr_debug") if save_debug else None

    log.info("OCR: start | dets=%d | debug_crops=%s", len(dets), save_debug)

    named = 0
    for d in dets:
        t_det = time.time()
        label = _ocr_label_for_det(base_img, d, debug_dir=debug_dir)
        d.name = label or ""

        if d.name:
            named += 1
            log.info("OCR OK   det=%d type=%s -> '%s'", d.idx, d.cls, d.name)
        else:
            log.warning("OCR FAIL det=%d type=%s (vazio)", d.idx, d.cls)

        log.debug(
            "OCR det=%d elapsed=%.1fms bbox=(%.1f,%.1f,%.1f,%.1f)",
            d.idx, (time.time() - t_det) * 1000.0, d.x1, d.y1, d.x2, d.y2
        )

    log.info("OCR: done | nomeados=%d/%d | elapsed=%.2fs", named, len(dets), time.time() - t0)
    return dets


# ======================================================
# WEIGHTS
# ======================================================
def find_best_weights_most_recent(runs_base: Path = Path("runs")) -> Path:
    log.info("Pesos: procurando best.pt em %s", str(runs_base))
    candidates = list(runs_base.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("❌ Não achei nenhum best.pt dentro de runs/. Treine primeiro.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    log.info("Pesos: selecionado %s", str(candidates[0]))
    return candidates[0]


# ======================================================
# DETECTION (assinatura compatível com seu app.py)
# ======================================================
def run_detection(
    image_path: Path,
    weights: Path | None = None,
    out_dir: Path = Path("runs_infer"),
    imgsz: int = IMGSZ,
    conf: float = CONF_THRES,
    iou: float = IOU_THRES,
) -> Tuple[List[Det], Dict[int, str], Path, Tuple[int, int]]:
    t0 = time.time()

    if not image_path.exists():
        raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")

    if weights is None:
        weights = find_best_weights_most_recent(Path("runs"))

    log.info(
        "YOLO: predict | image=%s | weights=%s | imgsz=%s conf=%s iou=%s",
        str(image_path), str(weights), imgsz, conf, iou
    )

    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        project=str(out_dir),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]
    names = r.names
    img_h, img_w = r.orig_shape[0], r.orig_shape[1]

    dets: List[Det] = []
    if r.boxes is not None and len(r.boxes) > 0:
        for i, b in enumerate(r.boxes):
            cls_id = int(b.cls[0])
            confv = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append(Det(
                idx=i,
                cls=names.get(cls_id, str(cls_id)),
                conf=confv,
                x1=x1, y1=y1, x2=x2, y2=y2,
                name="",
            ))

    saved_pred_dir = out_dir / "pred"

    log.info(
        "YOLO: done | dets=%d | img_w=%d img_h=%d | pred_dir=%s | elapsed=%.2fs",
        len(dets), img_w, img_h, str(saved_pred_dir), time.time() - t0
    )

    return dets, names, saved_pred_dir, (img_w, img_h)


# ======================================================
# GRAPH
# ======================================================
def assign_layers(dets: List[Det], img_w: int, n_layers: int = N_LAYERS) -> Dict[int, int]:
    layer_w = img_w / float(n_layers)
    layers: Dict[int, int] = {}
    for d in dets:
        layer = int(d.cx // layer_w)
        layer = max(0, min(n_layers - 1, layer))
        layers[d.idx] = layer
    log.debug("Layers: %s", layers)
    return layers


def build_edges(
    dets: List[Det],
    allowed: Dict[str, List[str]] = ALLOWED,
    out_degree: Dict[str, int] = OUT_DEGREE,
) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    dets_sorted = sorted(dets, key=lambda d: d.cx)

    for src in dets_sorted:
        allowed_targets = allowed.get(src.cls, [])
        k = out_degree.get(src.cls, 1)
        if k <= 0 or not allowed_targets:
            continue

        candidates = [dst for dst in dets_sorted if dst.cx > src.cx and dst.cls in allowed_targets]
        candidates.sort(key=lambda d: (d.cx - src.cx))

        for dst in candidates[:k]:
            edges.append((src.idx, dst.idx))

    edges = list(dict.fromkeys(edges))
    log.info("Edges: %d", len(edges))
    log.debug("Edges list: %s", edges)
    return edges


def export_graph_json(
    dets: List[Det],
    edges: List[Tuple[int, int]],
    layers: Dict[int, int],
    out_dir: Path = Path("runs_graph"),
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_json = []
    for d in dets:
        nodes_json.append({
            "id": d.idx,
            "type": d.cls,
            "name": d.name or "",
            "conf": float(d.conf),
            "layer": int(layers.get(d.idx, 0)),
            "bbox": [d.x1, d.y1, d.x2, d.y2],
            "center": [d.cx, d.cy],
        })

    edges_json = [{"src": int(a), "dst": int(b)} for a, b in edges]

    nodes_path = out_dir / "nodes.json"
    edges_path = out_dir / "edges.json"

    nodes_path.write_text(json.dumps(nodes_json, indent=2, ensure_ascii=False), encoding="utf-8")
    edges_path.write_text(json.dumps(edges_json, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Graph JSON: %s | %s", str(nodes_path), str(edges_path))
    return nodes_path, edges_path
