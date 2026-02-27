import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# -----------------------------
# Config
# -----------------------------
CLASSES = ["user", "server", "database", "api", "queue", "external", "storage"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

W, H = 1280, 720
BG_COLOR = (255, 255, 255)

DEFAULT_ICON_SIZE_RANGE = (64, 120)

# Where icons are stored:
# icons/
#   user/*.png
#   server/*.png
#   ...
ICON_ROOT = "icons"

# Output dataset folder
OUT_ROOT = Path("dataset_yolo")
IMAGES_DIR = OUT_ROOT / "images"
LABELS_DIR = OUT_ROOT / "labels"

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Set a seed for reproducibility (optional)
RANDOM_SEED = 42


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    for split in SPLIT_RATIOS.keys():
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)


def list_icons(icon_root: str, cls: str):
    p = Path(icon_root) / cls
    if not p.exists():
        return []
    files = [f for f in p.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    return files


def pick_icon(icon_root: str, cls: str) -> Image.Image:
    files = list_icons(icon_root, cls)
    if not files:
        raise FileNotFoundError(
            f"No icons found for class '{cls}' in {Path(icon_root)/cls}. "
            "Add PNG/JPG icons per class."
        )
    path = random.choice(files)
    return Image.open(path).convert("RGBA")


def draw_arrow(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, width=3):
    draw.line((x1, y1, x2, y2), width=width, fill=(0, 0, 0))
    # arrow head
    ang = np.arctan2(y2 - y1, x2 - x1)
    L = 14
    a1 = ang + np.pi * 3 / 4
    a2 = ang - np.pi * 3 / 4
    p1 = (x2 + L * np.cos(a1), y2 + L * np.sin(a1))
    p2 = (x2 + L * np.cos(a2), y2 + L * np.sin(a2))
    draw.polygon([(x2, y2), p1, p2], fill=(0, 0, 0))


def yolo_line(class_id, x, y, w, h):
    # x,y,w,h in pixels
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    wn = w / W
    hn = h / H
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def add_noise_and_artifacts(img: Image.Image) -> Image.Image:
    # Add light gaussian noise
    arr = np.array(img).astype(np.int16)
    noise_sigma = random.uniform(0, 8)
    noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)

    # Optional blur
    if random.random() < 0.25:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    # Small rotation (simulate screenshot skew)
    if random.random() < 0.20:
        angle = random.uniform(-2.0, 2.0)
        out = out.rotate(angle, expand=False, fillcolor=BG_COLOR)

    return out


def safe_font(size=16):
    # Tries to load a common font; falls back to default.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


# -----------------------------
# Templates (graphs) -> nodes + edges
# nodes: list of (class_name, label)
# edges: list of (src_index, dst_index)
# -----------------------------
def template_three_tier():
    nodes = [
        ("user", "User"),
        ("api", random.choice(["API", "Gateway", "Auth API"])),
        ("database", random.choice(["DB", "PostgreSQL", "MySQL"])),
    ]
    edges = [(0, 1), (1, 2)]
    return nodes, edges


def template_microservices():
    nodes = [("user", "User"), ("api", "API Gateway")]
    # 2-5 services
    n_services = random.randint(2, 5)
    for i in range(n_services):
        nodes.append(("server", f"svc-{i+1}"))
    nodes.append(("database", random.choice(["DB", "Orders DB", "Users DB"])))
    nodes.append(("queue", random.choice(["Queue", "Kafka", "RabbitMQ"])))

    # edges: user->gateway, gateway->services, services->db, services->queue
    edges = [(0, 1)]
    service_indices = list(range(2, 2 + n_services))
    for s in service_indices:
        edges.append((1, s))
        if random.random() < 0.8:
            edges.append((s, 2 + n_services))  # -> DB
        if random.random() < 0.6:
            edges.append((s, 3 + n_services))  # -> Queue
    return nodes, edges


def template_event_driven():
    nodes = [
        ("external", "Partner"),
        ("api", "Ingest API"),
        ("queue", random.choice(["Kafka", "Queue"])),
        ("server", "Consumer"),
        ("database", "DB"),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return nodes, edges


TEMPLATES = [template_three_tier, template_microservices, template_event_driven]


def layout_layered(nodes):
    """
    Very simple layered layout:
    - Place nodes in left-to-right order with jitter.
    """
    margin_x = 120
    n = len(nodes)
    step_x = (W - 2 * margin_x) // max(1, (n - 1))
    base_y = random.randint(200, 420)

    positions = []
    for i in range(n):
        x = margin_x + i * step_x + random.randint(-35, 35)
        y = base_y + random.randint(-45, 45)
        positions.append((x, y))
    return positions


def render_diagram(nodes, edges):
    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = safe_font(size=random.randint(14, 18))

    positions = layout_layered(nodes)

    annotations = []  # (class_name, x, y, w, h)
    centers = []

    # Render nodes
    for i, (cls, label) in enumerate(nodes):
        icon = pick_icon(ICON_ROOT, cls)

        size = random.randint(*DEFAULT_ICON_SIZE_RANGE)
        icon = icon.resize((size, size), Image.Resampling.LANCZOS)

        x, y = positions[i]

        # Draw optional light box behind
        if random.random() < 0.35:
            pad = random.randint(6, 12)
            draw.rounded_rectangle(
                (x - pad, y - pad, x + size + pad, y + size + pad),
                radius=12,
                outline=(0, 0, 0),
                width=2,
                fill=(255, 255, 255),
            )

        canvas.paste(icon, (x, y), icon)

        # label
        tx = x
        ty = y + size + random.randint(6, 10)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

        annotations.append((cls, x, y, size, size))
        centers.append((x + size // 2, y + size // 2))

    # Render edges (arrows)
    for (src, dst) in edges:
        x1, y1 = centers[src]
        x2, y2 = centers[dst]
        draw_arrow(draw, x1, y1, x2, y2, width=random.randint(2, 4))

    # Augmentations
    canvas = add_noise_and_artifacts(canvas)
    return canvas, annotations


def write_data_yaml():
    yaml_path = OUT_ROOT / "data.yaml"
    content = f"""path: {OUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

names:
"""
    for i, c in enumerate(CLASSES):
        content += f"  {i}: {c}\n"
    yaml_path.write_text(content, encoding="utf-8")


def make_splits(n: int, seed: int = 42):
    """
    Deterministic split based on shuffled indices.
    Guarantees at least 1 image in val when n>=2.
    """
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)

    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    # remainder goes to test
    n_test = n - n_train - n_val

    # Guarantee at least one val if possible
    if n >= 2 and n_val == 0:
        n_val = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_test = max(0, n_test - 1)

    train_ids = set(idxs[:n_train])
    val_ids = set(idxs[n_train : n_train + n_val])
    test_ids = set(idxs[n_train + n_val :])

    return train_ids, val_ids, test_ids


def main(num_images=2000, clean=True):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if clean and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    ensure_dirs()
    write_data_yaml()

    train_ids, val_ids, test_ids = make_splits(num_images, RANDOM_SEED)

    for idx in range(num_images):
        # pick template
        tpl = random.choice(TEMPLATES)
        nodes, edges = tpl()

        img, ann = render_diagram(nodes, edges)

        if idx in train_ids:
            split = "train"
        elif idx in val_ids:
            split = "val"
        else:
            split = "test"

        img_name = f"{idx:06d}.png"
        lbl_name = f"{idx:06d}.txt"

        img_path = IMAGES_DIR / split / img_name
        lbl_path = LABELS_DIR / split / lbl_name

        img.save(img_path)

        # YOLO labels
        lines = []
        for cls, x, y, w, h in ann:
            class_id = CLASS_TO_ID[cls]
            lines.append(yolo_line(class_id, x, y, w, h))
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

    print("✅ Dataset gerado em:", OUT_ROOT.resolve())
    print("✅ YAML:", (OUT_ROOT / "data.yaml").resolve())
    print("👉 Próximo passo: rode o script trainer.py")


def generate_dataset(num_images: int = 200, clean: bool = True) -> str:
    """
    Gera o dataset YOLO em dataset_yolo/.
    Retorna o caminho do data.yaml.
    """
    # aqui dentro você chama seu main(num_images=..., clean=...)
    # mantendo seu código praticamente igual.
    main(num_images=num_images, clean=clean)  # <- seu main original
    return str((OUT_ROOT / "data.yaml").resolve())
