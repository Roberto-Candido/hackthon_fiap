# trainer.py
from __future__ import annotations

from pathlib import Path
import torch

def train_yolo(
    data_yaml: str,
    model_ckpt: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    project: str = "runs",
    name: str = "yolo_components",
) -> str:
    """
    Treina YOLO com Ultralytics e retorna o caminho do best.pt (se existir).
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Falha ao importar ultralytics. Instale com: pip install ultralytics") from e

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml}")

    model = YOLO(model_ckpt)

    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data=str(data_path),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        workers=workers,
        device=device,
        project=project,
        name=name,
    )

    # Ultralytics geralmente salva em runs/detect/<name>/weights/best.pt
    # Mas pode variar; então buscamos o mais recente:
    best = find_best_weights_most_recent(Path(project))
    return str(best.resolve())


def find_best_weights_most_recent(runs_base: Path) -> Path:
    candidates = list(runs_base.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("Não achei best.pt após treino. Veja logs do ultralytics.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
