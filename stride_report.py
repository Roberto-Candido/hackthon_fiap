from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Node:
    id: int
    type: str          # user, server, database, api, queue, external, storage
    name: str = ""     # opcional, pode vir vazio
    conf: float = 1.0  # confiança YOLO (se tiver)
    layer: int = 0     # opcional (se você já calcula)

@dataclass
class Edge:
    src: int
    dst: int

@dataclass
class Threat:
    target_kind: str       # "node" | "flow"
    target_id: str         # node:<id> or flow:<src>-><dst>
    stride: str            # S/T/R/I/D/E
    title: str
    description: str
    vulnerabilities: List[str]
    mitigations: List[str]
    score: int             # heurístico
    evidence: List[str]    # rastreabilidade


# -----------------------------
# STRIDE knowledge base (MVP)
# -----------------------------
STRIDE_LABELS = {
    "S": "Spoofing",
    "T": "Tampering",
    "R": "Repudiation",
    "I": "Information Disclosure",
    "D": "Denial of Service",
    "E": "Elevation of Privilege",
}

NODE_STRIDE = {
    "user":     ["S", "R"],
    "external": ["S", "R"],
    "api":      ["S", "T", "R", "I", "D", "E"],
    "server":   ["S", "T", "R", "I", "D", "E"],
    "queue":    ["T", "I", "D"],
    "database": ["T", "I", "D"],
    "storage":  ["T", "I", "D"],
}

NODE_THREATS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("api", "S"): {
        "title": "Spoofing de identidade no endpoint",
        "desc": "Um agente pode se passar por usuário/serviço e acessar endpoints se autenticação/autorização forem fracas.",
        "vulns": ["Ausência de autenticação forte", "Tokens previsíveis", "Chaves expostas", "Falta de mTLS entre serviços"],
        "mitig": ["OIDC/OAuth2 bem configurado", "mTLS entre serviços", "Rotação de chaves", "Validação de issuer/audience"],
    },
    ("api", "T"): {
        "title": "Adulteração de requisições/respostas",
        "desc": "Dados podem ser alterados em trânsito ou por validação insuficiente no backend.",
        "vulns": ["Sem TLS", "Validação fraca de payload", "Falta de assinatura/nonce"],
        "mitig": ["TLS obrigatório", "Validação de schema", "Assinatura/HMAC para integrações críticas", "Idempotency keys"],
    },
    ("api", "R"): {
        "title": "Repúdio de operações",
        "desc": "Usuários/serviços podem negar ações se não houver logs e trilha de auditoria confiáveis.",
        "vulns": ["Logs incompletos", "Sem correlação de requisições", "Logs alteráveis"],
        "mitig": ["Audit logs imutáveis", "Correlation IDs", "Centralização de logs (SIEM)", "Assinatura de logs"],
    },
    ("api", "I"): {
        "title": "Exposição de informações sensíveis",
        "desc": "Dados sensíveis podem vazar via respostas, headers, logs ou erros.",
        "vulns": ["Verbose errors", "Logs com PII", "Exposição de secrets", "CORS mal configurado"],
        "mitig": ["Sanitização de logs", "Tratamento de erro seguro", "Secrets manager", "Revisão de CORS e headers"],
    },
    ("api", "D"): {
        "title": "Negação de serviço (DoS) no gateway/API",
        "desc": "Sobrecarga por alto volume, payloads grandes ou endpoints caros.",
        "vulns": ["Sem rate limit", "Sem quotas", "Sem circuit breaker", "Queries caras sem proteção"],
        "mitig": ["Rate limiting", "WAF", "Timeouts", "Circuit breaker", "Cache", "Filas/assíncrono"],
    },
    ("api", "E"): {
        "title": "Escalonamento de privilégio via falhas de autorização",
        "desc": "Acesso indevido a recursos por falhas em RBAC/ABAC e checagens de permissão.",
        "vulns": ["IDOR", "Checagem só no front", "Permissões por rota mal definidas"],
        "mitig": ["RBAC/ABAC no backend", "Policy enforcement", "Testes de autorização", "Princípio do menor privilégio"],
    },

    ("server", "S"): {
        "title": "Impersonação entre serviços",
        "desc": "Um serviço malicioso pode se passar por outro se a identidade não for verificada.",
        "vulns": ["Sem mTLS", "Chaves compartilhadas", "Service-to-service auth ausente"],
        "mitig": ["mTLS", "Identidade por workload", "Rotação de credenciais", "Service mesh (opcional)"],
    },
    ("server", "T"): {
        "title": "Adulteração de dados/processamento",
        "desc": "Código, configs ou dados processados podem ser alterados indevidamente.",
        "vulns": ["Config exposta", "Supply chain", "Sem controle de integridade"],
        "mitig": ["Assinatura de artefatos", "Hardening", "Controle de acesso a configs", "SAST/DAST", "SBOM"],
    },
    ("server", "R"): {
        "title": "Repúdio por falta de auditoria",
        "desc": "Sem logs/auditoria confiáveis, ações podem ser negadas.",
        "vulns": ["Logs locais", "Sem correlação", "Sem retenção"],
        "mitig": ["Logs centralizados", "Retention", "Correlation IDs", "Trilha de auditoria"],
    },
    ("server", "I"): {
        "title": "Vazamento de informações em memória/arquivos/logs",
        "desc": "Dados sensíveis podem vazar em logs, dumps, arquivos temporários.",
        "vulns": ["Logs com PII", "Dumps em produção", "Secrets em variáveis"],
        "mitig": ["Mascaramento", "Desabilitar dumps", "Secrets manager", "Least privilege"],
    },
    ("server", "D"): {
        "title": "DoS por esgotamento de recursos",
        "desc": "CPU/memória/threads podem ser exauridos por requisições maliciosas.",
        "vulns": ["Sem limites", "Sem timeouts", "Sem backpressure"],
        "mitig": ["Timeouts", "Bulkheads", "Autoscaling", "Backpressure", "Filas"],
    },
    ("server", "E"): {
        "title": "Elevação de privilégio no host/serviço",
        "desc": "Exploração pode permitir privilégios acima do esperado.",
        "vulns": ["Containers privilegiados", "Permissões excessivas", "Dependências vulneráveis"],
        "mitig": ["Hardening", "Privilégios mínimos", "Patch management", "Execução não-root"],
    },

    ("database", "T"): {
        "title": "Adulteração de dados no datastore",
        "desc": "Alteração indevida de dados armazenados por acessos não autorizados ou SQL injection.",
        "vulns": ["Credenciais fracas", "SQL injection", "Permissões excessivas"],
        "mitig": ["Least privilege", "Rotação de credenciais", "Prepared statements", "Auditoria de alterações"],
    },
    ("database", "I"): {
        "title": "Exfiltração de dados sensíveis",
        "desc": "Vazamento de dados por acessos indevidos, backups expostos ou consultas não controladas.",
        "vulns": ["Backups públicos", "Sem criptografia", "Acesso amplo à rede"],
        "mitig": ["Criptografia at-rest", "Network segmentation", "Controle de acesso", "DLP (opcional)"],
    },
    ("database", "D"): {
        "title": "DoS no banco",
        "desc": "Consultas pesadas, lock contention e saturação podem derrubar o banco.",
        "vulns": ["Sem limites de query", "Sem pool", "Sem índices"],
        "mitig": ["Pool", "Índices", "Timeouts", "Rate limit upstream", "Read replicas (opcional)"],
    },

    ("storage", "T"): {
        "title": "Adulteração de artefatos/objetos",
        "desc": "Objetos armazenados podem ser modificados por acessos indevidos.",
        "vulns": ["ACLs permissivas", "URLs assinadas mal geridas"],
        "mitig": ["ACL mínimo", "Versionamento", "Checksum/ETag", "Assinatura/URLs temporárias"],
    },
    ("storage", "I"): {
        "title": "Exposição de arquivos/objetos",
        "desc": "Arquivos podem ser acessados publicamente ou por credenciais vazadas.",
        "vulns": ["Bucket público", "Chaves expostas"],
        "mitig": ["Bloqueio de acesso público", "Secrets manager", "Logs de acesso", "Criptografia"],
    },
    ("storage", "D"): {
        "title": "DoS por saturação de armazenamento",
        "desc": "Upload abusivo pode aumentar custo e indisponibilidade.",
        "vulns": ["Sem quotas", "Sem validação de tipo/tamanho"],
        "mitig": ["Quotas", "Validação de tamanho", "Rate limits", "Monitoramento"],
    },

    ("queue", "T"): {
        "title": "Adulteração de mensagens",
        "desc": "Mensagens podem ser alteradas/forjadas se não houver integridade e autenticação.",
        "vulns": ["Sem auth", "Sem assinatura de payload"],
        "mitig": ["Authn/Authz", "Assinatura/HMAC", "Schema validation", "DLQ"],
    },
    ("queue", "I"): {
        "title": "Vazamento em mensagens",
        "desc": "Mensagens podem carregar PII e vazar se transporte/armazenamento não forem protegidos.",
        "vulns": ["Sem criptografia", "Tópicos acessíveis"],
        "mitig": ["Criptografia in-transit", "Criptografia at-rest", "Controle de acesso"],
    },
    ("queue", "D"): {
        "title": "DoS por backlog/poison messages",
        "desc": "Mensagens maliciosas podem travar consumidores e gerar backlog.",
        "vulns": ["Sem DLQ", "Sem retry controlado"],
        "mitig": ["DLQ", "Circuit breaker", "Backoff", "Observabilidade"],
    },
}

FLOW_THREATS: Dict[str, Dict[str, Any]] = {
    "S": {
        "title": "Spoofing/MITM no fluxo",
        "desc": "Um atacante pode se passar por uma das pontas do fluxo se não houver autenticação mútua e canal seguro.",
        "vulns": ["Sem TLS/mTLS", "DNS spoofing", "Tokens reutilizáveis"],
        "mitig": ["TLS", "mTLS (interno)", "Pinning (casos específicos)", "Rotação de tokens"],
    },
    "T": {
        "title": "Tampering em trânsito",
        "desc": "Dados em trânsito podem ser alterados sem integridade/autenticidade.",
        "vulns": ["Sem TLS", "Sem assinatura", "Sem nonce/idempotência"],
        "mitig": ["TLS", "Assinatura/HMAC (integrações críticas)", "Idempotency keys"],
    },
    "R": {
        "title": "Repúdio de transações",
        "desc": "Sem trilha de auditoria e correlação, ações no fluxo podem ser negadas.",
        "vulns": ["Logs sem correlação", "Sem timestamps confiáveis"],
        "mitig": ["Correlation IDs", "Audit logs", "Time sync", "Imutabilidade de logs"],
    },
    "I": {
        "title": "Information disclosure no fluxo",
        "desc": "Dados sensíveis podem ser expostos em trânsito se o canal e payload não forem protegidos.",
        "vulns": ["Sem TLS", "Payload com PII sem proteção"],
        "mitig": ["TLS", "Minimização de dados", "Criptografia end-to-end (casos críticos)"],
    },
    "D": {
        "title": "DoS no fluxo",
        "desc": "O fluxo pode ser saturado com alto volume, causando indisponibilidade.",
        "vulns": ["Sem rate limit", "Sem timeouts", "Sem filas/circuit breaker"],
        "mitig": ["Rate limit", "Timeouts", "Circuit breaker", "Filas"],
    },
}


# -----------------------------
# Heuristics (boundary + scoring)
# -----------------------------
def is_boundary_crossing(a: Node, b: Node) -> bool:
    ext = {"user", "external"}
    internal = {"api", "server", "queue", "database", "storage"}
    return (a.type in ext and b.type in internal) or (b.type in ext and a.type in internal)

def base_score_for_stride(stride: str) -> int:
    return {"S": 6, "T": 7, "R": 4, "I": 8, "D": 7, "E": 9}.get(stride, 5)

def bump_for_node_type(node_type: str) -> int:
    if node_type in {"database", "storage"}:
        return 2
    if node_type in {"api", "server"}:
        return 1
    return 0

def bump_for_boundary(crossing: bool) -> int:
    return 2 if crossing else 0

def clamp_score(x: int) -> int:
    return max(1, min(10, x))


# -----------------------------
# Helpers: labels bonitos
# -----------------------------
def node_display(n: Node) -> str:
    """
    Ex: server(Svc 2) / api(APTGateway) / database(Orders DB)
    """
    nm = (n.name or "").strip()
    if not nm:
        return f"{n.type}(sem nome)"
    return f"{n.type}({nm})"

def flow_display(src: Node, dst: Node) -> str:
    return f"{node_display(src)} -> {node_display(dst)}"

def threat_target_display(t: Threat, id2: Dict[int, Node]) -> str:
    """
    Mantém o ID técnico, mas adiciona nome:
      node:3 database(Orders DB)
      flow:2->5 user(APTGatevay) -> api(APTGaleway)
    """
    if t.target_id.startswith("node:"):
        try:
            nid = int(t.target_id.split("node:")[1])
        except Exception:
            return t.target_id
        n = id2.get(nid)
        if not n:
            return t.target_id
        return f"{t.target_id} {node_display(n)}"

    if t.target_id.startswith("flow:"):
        try:
            raw = t.target_id.split("flow:")[1]
            a_str, b_str = raw.split("->")
            a_id = int(a_str)
            b_id = int(b_str)
        except Exception:
            return t.target_id
        a = id2.get(a_id)
        b = id2.get(b_id)
        if not a or not b:
            return t.target_id
        return f"{t.target_id} {flow_display(a, b)}"

    return t.target_id


# -----------------------------
# Core engine
# -----------------------------
def generate_stride_threats(
    nodes: List[Node],
    edges: List[Edge],
) -> List[Threat]:
    id2 = {n.id: n for n in nodes}
    threats: List[Threat] = []

    # Node threats
    for n in nodes:
        for s in NODE_STRIDE.get(n.type, []):
            kb = NODE_THREATS.get((n.type, s))
            if not kb:
                kb = {
                    "title": f"{STRIDE_LABELS.get(s, s)} em {n.type}",
                    "desc": f"Risco {STRIDE_LABELS.get(s, s)} aplicável ao componente do tipo {n.type}.",
                    "vulns": [],
                    "mitig": [],
                }
            score = clamp_score(base_score_for_stride(s) + bump_for_node_type(n.type))
            threats.append(Threat(
                target_kind="node",
                target_id=f"node:{n.id}",
                stride=s,
                title=kb["title"],
                description=kb["desc"],
                vulnerabilities=kb.get("vulns", []),
                mitigations=kb.get("mitig", []),
                score=score,
                evidence=[
                    f"node_type={n.type}",
                    f"node_name={n.name or ''}",
                    f"conf={n.conf:.2f}",
                ],
            ))

    # Flow threats
    for e in edges:
        a = id2.get(e.src)
        b = id2.get(e.dst)
        if not a or not b:
            continue

        crossing = is_boundary_crossing(a, b)
        strides = ["S", "T", "I", "D", "R"] if crossing else ["T", "I", "D"]

        for s in strides:
            kb = FLOW_THREATS.get(s)
            if not kb:
                continue
            score = clamp_score(base_score_for_stride(s) + bump_for_boundary(crossing))
            threats.append(Threat(
                target_kind="flow",
                target_id=f"flow:{e.src}->{e.dst}",
                stride=s,
                title=kb["title"],
                description=kb["desc"],
                vulnerabilities=kb.get("vulns", []),
                mitigations=kb.get("mitig", []),
                score=score,
                evidence=[
                    f"src={node_display(a)}",
                    f"dst={node_display(b)}",
                    f"boundary_crossing={crossing}",
                ],
            ))

    threats.sort(key=lambda t: (t.score, t.stride), reverse=True)
    return threats


def render_markdown_report(
    nodes: List[Node],
    edges: List[Edge],
    threats: List[Threat],
    title: str = "Relatório de Modelagem de Ameaças (STRIDE)",
    assumptions: Optional[List[str]] = None,
) -> str:
    assumptions = assumptions or [
        "Componentes foram inferidos por detecção visual (YOLO) a partir do diagrama.",
        "Fluxos (arestas) podem ter sido inferidos automaticamente e podem exigir revisão.",
        "Fronteira de confiança foi inferida por heurísticas simples (user/external → interno).",
        "Ameaças e contramedidas foram geradas por base de regras determinísticas (MVP).",
    ]

    counts: Dict[str, int] = {}
    for n in nodes:
        counts[n.type] = counts.get(n.type, 0) + 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    id2 = {n.id: n for n in nodes}

    md: List[str] = []
    md.append(f"# {title}")
    md.append(f"- Gerado em: **{now}**")
    md.append("")

    md.append("## 1. Resumo do sistema")
    md.append(f"- Total de componentes detectados: **{len(nodes)}**")
    md.append(f"- Total de fluxos inferidos: **{len(edges)}**")
    md.append("- Componentes por tipo:")
    for k in sorted(counts.keys()):
        md.append(f"  - **{k}**: {counts[k]}")
    md.append("")

    md.append("## 2. Assunções e limitações")
    for a in assumptions:
        md.append(f"- {a}")
    md.append("")

    md.append("## 3. Modelo inferido (nós)")
    md.append("| id | tipo | nome | conf | layer |")
    md.append("|---:|------|------|------:|------:|")
    for n in nodes:
        name = n.name if n.name else "(sem nome)"
        md.append(f"| {n.id} | {n.type} | {name} | {n.conf:.2f} | {n.layer} |")
    md.append("")

    md.append("## 4. Modelo inferido (fluxos)")
    md.append("| fluxo | origem | destino | cruza fronteira? |")
    md.append("|------|--------|---------|------------------|")
    for e in edges:
        a = id2.get(e.src)
        b = id2.get(e.dst)
        if not a or not b:
            continue
        cross = is_boundary_crossing(a, b)
        md.append(f"| {e.src}→{e.dst} | {node_display(a)} | {node_display(b)} | {'SIM' if cross else 'NÃO'} |")
    md.append("")

    md.append("## 5. Ameaças identificadas (priorizadas)")
    md.append("> Pontuação de 1 a 10 é heurística para priorização no MVP.")
    md.append("")
    md.append("| score | alvo | STRIDE | título |")
    md.append("|------:|------|--------|--------|")
    for t in threats[:80]:
        target_pretty = threat_target_display(t, id2)
        md.append(f"| {t.score} | {target_pretty} | {t.stride} ({STRIDE_LABELS.get(t.stride)}) | {t.title} |")
    md.append("")

    md.append("## 6. Detalhamento das principais ameaças")
    top = threats[:15]
    for i, t in enumerate(top, 1):
        md.append(f"### {i}. [{t.score}/10] {t.title}")
        md.append(f"- **Alvo:** `{threat_target_display(t, id2)}` ({t.target_kind})")
        md.append(f"- **Categoria:** {t.stride} — {STRIDE_LABELS.get(t.stride)}")
        md.append(f"- **Descrição:** {t.description}")

        if t.vulnerabilities:
            md.append("- **Vulnerabilidades comuns:**")
            for v in t.vulnerabilities:
                md.append(f"  - {v}")

        if t.mitigations:
            md.append("- **Contramedidas sugeridas:**")
            for m in t.mitigations:
                md.append(f"  - {m}")

        if t.evidence:
            md.append("- **Evidências/heurísticas aplicadas:**")
            for ev in t.evidence:
                md.append(f"  - {ev}")

        md.append("")

    md.append("## 7. Recomendações rápidas (MVP)")
    md.append("- Garantir **TLS/mTLS** em fluxos críticos e especialmente nos que cruzam fronteira.")
    md.append("- Implementar **autenticação e autorização** robustas (RBAC/ABAC) em APIs e serviços.")
    md.append("- Aplicar **logging/auditoria** com correlação (Correlation ID) e retenção adequada.")
    md.append("- Proteger segredos com **Secrets Manager** e rotação de credenciais.")
    md.append("- Mitigar DoS com **rate limiting, timeouts, circuit breaker** e filas quando aplicável.")
    md.append("")

    return "\n".join(md)


def generate_report_artifacts(
    nodes: List[Node],
    edges: List[Edge],
    out_dir: str,
    title: str = "Relatório de Modelagem de Ameaças (STRIDE)",
) -> Dict[str, str]:
    import os
    os.makedirs(out_dir, exist_ok=True)

    threats = generate_stride_threats(nodes, edges)
    md = render_markdown_report(nodes, edges, threats, title=title)

    md_path = f"{out_dir}/report.md"
    json_path = f"{out_dir}/report.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    payload = {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "threats": [asdict(t) for t in threats],
        "meta": {"title": title, "generated_at": datetime.now().isoformat()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {"md": md_path, "json": json_path}
