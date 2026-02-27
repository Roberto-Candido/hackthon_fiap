from pathlib import Path
from ultralytics import YOLO


def find_best_weights(
    base_runs="runs",
    task="detect",
    project="runs_arch",
    name="yolo_components",
) -> Path:
    """
    Caminho real do Ultralytics:
    runs/detect/<project>/<name>/weights/best.pt
    """
    p = (
        Path(base_runs)
        / task
        / project
        / name
        / "weights"
        / "best.pt"
    )

    if p.exists():
        return p

    # fallback: procura qualquer best.pt dentro de runs/
    candidates = list(Path(base_runs).rglob("best.pt"))
    if candidates:
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    raise FileNotFoundError(
        "❌ Não achei nenhum best.pt em runs/. "
        "Confirme se o treino terminou corretamente."
    )


def run_inference(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")

    weights = find_best_weights()
    print(f"✅ Usando pesos: {weights}")

    model = YOLO(str(weights))

    out_dir = Path("runs_infer")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        save=True,
        project=str(out_dir),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]

    print("\n📦 Detecções:")
    if r.boxes is None or len(r.boxes) == 0:
        print("  (nenhuma detecção)")
    else:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cls_name = names.get(cls_id, str(cls_id))
            print(
                f"  - {cls_name:10s} | conf={conf:.3f} | "
                f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )

    print(f"\n🖼️ Imagem salva em: {out_dir / 'pred'}")
    print("✅ Inferência concluída.")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 👇 AJUSTA SÓ ISSO
    IMAGE_NAME = "000057.png"
    IMAGE_PATH = Path("imagens_teste") / IMAGE_NAME
    # Exemplo alternativo:
    # IMAGE_PATH = Path("dataset_yolo/images/val/000001.png")

    run_inference(IMAGE_PATH)
