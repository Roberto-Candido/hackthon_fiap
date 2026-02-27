# Modelagem de Ameaças com IA — STRIDE a partir de Diagramas de Arquitetura

Este projeto implementa um **MVP (Minimum Viable Product)** para avaliar a viabilidade do uso de **Inteligência Artificial** na **modelagem automática de ameaças**, utilizando a metodologia **STRIDE**, a partir de **diagramas de arquitetura de software em formato de imagem**.

A solução combina:
- Visão computacional (YOLO) para detecção de componentes arquiteturais;
- Processamento de imagem e OCR para enriquecimento semântico;
- Construção de um grafo arquitetural;
- Aplicação da metodologia STRIDE;
- Geração automática de relatório de ameaças.

---

## Visão geral do pipeline

1. Geração de dataset supervisionado (YOLO)
2. Treinamento do modelo de detecção
3. Inferência em diagramas reais
4. Construção do grafo (componentes e fluxos)
5. Aplicação do STRIDE
6. Geração do relatório de ameaças

---

## Requisitos

### Software
- Python 3.10 ou superior
- pip
- (Opcional) GPU com CUDA para acelerar o treinamento

### Bibliotecas principais
- ultralytics
- streamlit
- torch
- pillow
- numpy
- easyocr
- matplotlib

---

## Instalação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

### 2. Crie e ative um ambiente virtual (recomendado)
```bash
python -m venv venv
```

Linux / macOS:
```bash
source venv/bin/activate
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

Caso não exista um `requirements.txt`:
```bash
pip install ultralytics streamlit torch pillow numpy easyocr matplotlib
```

---

## Estrutura do projeto

```
.
├── app.py
├── arch_pipeline.py
├── stride_report.py
├── trainer.py
├── dataset_generator.py
├── icons/
├── dataset_yolo/
├── runs/
├── runs_infer/
├── runs_graph/
├── runs_stride/
└── README.md
```

---

## Classes de componentes suportadas

- user
- external
- api
- server
- database
- queue
- storage

Cada classe deve possuir ícones correspondentes na pasta `icons/`.

---

## Como rodar o projeto

### Etapa 1 — Preparar os ícones

Organize os ícones da seguinte forma:

```
icons/
├── user/
├── api/
├── server/
├── database/
├── queue/
├── external/
└── storage/
```

---

### Etapa 2 — Gerar o dataset

Via interface:
```bash
streamlit run app.py
```

Menu **Dataset** → **Gerar Dataset**

Ou via código:
```python
from dataset_generator import generate_dataset
generate_dataset(num_images=500, clean=True)
```

---

### Etapa 3 — Treinar o modelo YOLO

Via interface Streamlit (menu Treinamento) ou via script:
```bash
python trainer.py
```

O arquivo `best.pt` será salvo automaticamente na pasta `runs/`.

---

### Etapa 4 — Inferência e STRIDE

1. Execute:
```bash
streamlit run app.py
```

2. Vá para **Inferência & STRIDE**
3. Faça upload de um diagrama (PNG/JPG)
4. Execute a inferência
5. Revise componentes e fluxos
6. Gere o relatório STRIDE

---

## Saídas geradas

### Inferência
- Imagem anotada com YOLO
- nodes.json
- edges.json

### Relatório
- runs_stride/report.md
- runs_stride/report.json

---

## Observações

- O OCR pode falhar dependendo da qualidade da imagem.
- A inferência de fluxos é heurística e pode exigir revisão manual.
- Projeto desenvolvido como MVP acadêmico.

---

## Objetivo do MVP

Demonstrar a viabilidade do uso de IA para:
- interpretar diagramas de arquitetura;
- estruturar modelos arquiteturais;
- aplicar STRIDE automaticamente;
- apoiar decisões de segurança de software.

---

## Autor

Projeto desenvolvido para fins acadêmicos no contexto de Segurança de Software.
