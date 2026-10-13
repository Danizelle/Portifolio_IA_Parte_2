"""
Projeto 2 - Busca não informada: robô de entrega em mapa com terrenos
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
Um robô móvel precisa ir da base (canto superior esquerdo) ao ponto de entrega
(canto inferior direito) de um mapa em grade 2-D. Cada célula é:
  * obstáculo (intransponível);
  * piso   - custo 1 para entrar;
  * grama  - custo 3 para entrar;
  * lama   - custo 8 para entrar.
O robô se move nas 4 direções (N, L, S, O). O agente NÃO usa nenhuma estimativa
da distância ao objetivo - apenas a definição do problema (estado inicial,
ações, modelo de transição, teste de objetivo e custo de passo).

Algoritmos (todos como busca em grafo):
  * Busca em largura (BFS)                    - fila FIFO;
  * Busca em profundidade (DFS)               - pilha LIFO;
  * Busca de custo uniforme (UCS / Dijkstra)  - fila de prioridade por g(n);
  * Aprofundamento iterativo (IDS)            - DFS com limite crescente.

Dois cenários: (A) todos os terrenos com custo 1; (B) custos reais (1/3/8).
Também é medido o crescimento do esforço com o tamanho do mapa.

Execução:  python p2_busca_nao_informada_robo.py
Saídas:    figs/p2_*.png  e  results/p2.json
"""

import heapq
import json
import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Códigos das células e custos de entrada
OBST, PISO, GRAMA, LAMA = -1, 0, 1, 2
CUSTO_REAL = {PISO: 1, GRAMA: 3, LAMA: 8}
CUSTO_UNIT = {PISO: 1, GRAMA: 1, LAMA: 1}
# Ordem fixa de geração dos sucessores: Norte, Leste, Sul, Oeste
MOVS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# --------------------------------------------------------------------------
# Geração do mapa
# --------------------------------------------------------------------------
def suavizar(m, k=2):
    """Média móvel 2-D simples, para que os terrenos formem manchas contínuas."""
    out = np.zeros_like(m)
    n, mm = m.shape
    for i in range(n):
        for j in range(mm):
            out[i, j] = m[max(0, i - k):i + k + 1, max(0, j - k):j + k + 1].mean()
    return out


def gerar_mapa(n, seed, dens_obst=0.22):
    """Cria um mapa n x n com manchas de grama/lama e obstáculos aleatórios.
    Repete a semente até existir caminho entre início e objetivo."""
    while True:
        rng = np.random.default_rng(seed)
        ruido = suavizar(rng.random((n, n)))
        mapa = np.full((n, n), PISO)
        q1, q2 = np.quantile(ruido, [0.45, 0.72])
        mapa[ruido > q1] = GRAMA
        mapa[ruido > q2] = LAMA
        mapa[rng.random((n, n)) < dens_obst] = OBST
        mapa[0, 0] = mapa[n - 1, n - 1] = PISO
        prob = Problema(mapa, CUSTO_UNIT)
        if bfs(prob)["caminho"] is not None:
            return mapa
        seed += 1000


class Problema:
    """Formulação do problema de busca (estado = (linha, coluna))."""

    def __init__(self, mapa, custos):
        self.mapa = mapa
        self.n = mapa.shape[0]
        self.inicio = (0, 0)
        self.objetivo = (self.n - 1, self.n - 1)
        self.custos = custos

    def acoes(self, s):
        """Sucessores válidos de s, na ordem N, L, S, O."""
        i, j = s
        for di, dj in MOVS:
            a, b = i + di, j + dj
            if 0 <= a < self.n and 0 <= b < self.n and self.mapa[a, b] != OBST:
                yield (a, b)

    def custo(self, s2):
        """Custo de passo = custo de entrar na célula de destino."""
        return self.custos[self.mapa[s2]]

    def custo_caminho(self, caminho):
        return sum(self.custo(s) for s in caminho[1:])


def reconstruir(pai, s):
    caminho = []
    while s is not None:
        caminho.append(s)
        s = pai[s]
    return caminho[::-1]


# --------------------------------------------------------------------------
# Algoritmos
# --------------------------------------------------------------------------
def bfs(p):
    """Largura: expande por camadas. Teste de objetivo na GERAÇÃO (como em
    Russell e Norvig), o que economiza uma camada inteira de expansões."""
    if p.inicio == p.objetivo:
        return {"caminho": [p.inicio], "expandidos": 0, "max_mem": 1}
    fronteira = deque([p.inicio])
    pai = {p.inicio: None}                 # 'pai' também funciona como alcançados
    expandidos, max_mem = 0, 1
    while fronteira:
        s = fronteira.popleft()
        expandidos += 1
        for f in p.acoes(s):
            if f not in pai:
                pai[f] = s
                if f == p.objetivo:
                    return {"caminho": reconstruir(pai, f),
                            "expandidos": expandidos, "max_mem": len(pai)}
                fronteira.append(f)
        max_mem = max(max_mem, len(pai))
    return {"caminho": None, "expandidos": expandidos, "max_mem": max_mem}


def dfs(p):
    """Profundidade (em grafo): pilha LIFO e conjunto de expandidos.
    Encontra um caminho rapidamente, mas sem garantia de otimalidade."""
    pilha = [p.inicio]
    pai = {p.inicio: None}
    expandidos_set = set()
    max_mem = 1
    while pilha:
        s = pilha.pop()
        if s in expandidos_set:
            continue
        if s == p.objetivo:
            return {"caminho": reconstruir(pai, s),
                    "expandidos": len(expandidos_set), "max_mem": max_mem}
        expandidos_set.add(s)
        # empilha em ordem inversa para que o Norte seja explorado primeiro
        for f in reversed(list(p.acoes(s))):
            if f not in expandidos_set:
                pai[f] = s
                pilha.append(f)
        max_mem = max(max_mem, len(pilha) + len(expandidos_set))
    return {"caminho": None, "expandidos": len(expandidos_set), "max_mem": max_mem}


def ucs(p):
    """Custo uniforme: fila de prioridade por g(n). Teste de objetivo na
    EXPANSÃO - necessário para garantir o caminho de menor custo."""
    g = {p.inicio: 0}
    pai = {p.inicio: None}
    fila = [(0, p.inicio)]
    expandidos = set()
    max_mem = 1
    while fila:
        custo, s = heapq.heappop(fila)
        if s in expandidos:
            continue
        if s == p.objetivo:
            return {"caminho": reconstruir(pai, s),
                    "expandidos": len(expandidos), "max_mem": max_mem}
        expandidos.add(s)
        for f in p.acoes(s):
            ng = custo + p.custo(f)
            if f not in expandidos and ng < g.get(f, float("inf")):
                g[f] = ng
                pai[f] = s
                heapq.heappush(fila, (ng, f))
        max_mem = max(max_mem, len(fila) + len(expandidos))
    return {"caminho": None, "expandidos": len(expandidos), "max_mem": max_mem}


def ids(p, limite_max=10_000):
    """Aprofundamento iterativo: repete uma busca em profundidade limitada com
    limite 0, 1, 2, ... até achar o objetivo. Para evitar a explosão de
    caminhos redundantes numa grade, cada iteração guarda a menor profundidade
    em que cada célula já foi visitada e corta ramos que chegam a ela por um
    caminho igual ou mais longo (tabela de transposição)."""
    expandidos_total = 0
    max_mem = 0
    for limite in range(limite_max):
        melhor_prof = {p.inicio: 0}
        caminho = [p.inicio]
        # pilha de iteradores: implementação iterativa da recursão
        pilha = [iter(p.acoes(p.inicio))] if limite > 0 else []
        if p.inicio == p.objetivo:
            return {"caminho": caminho, "expandidos": 0, "max_mem": 1,
                    "iteracoes": 1}
        expandidos_total += 1 if limite > 0 else 0
        while pilha:
            try:
                f = next(pilha[-1])
            except StopIteration:
                pilha.pop()
                caminho.pop()
                continue
            prof = len(caminho)
            if melhor_prof.get(f, float("inf")) <= prof:
                continue
            melhor_prof[f] = prof
            caminho.append(f)
            if f == p.objetivo:
                return {"caminho": list(caminho),
                        "expandidos": expandidos_total,
                        "max_mem": max(max_mem, len(melhor_prof)),
                        "iteracoes": limite + 1}
            if prof < limite:
                expandidos_total += 1
                pilha.append(iter(p.acoes(f)))
            else:
                caminho.pop()
        max_mem = max(max_mem, len(melhor_prof))
    return {"caminho": None, "expandidos": expandidos_total, "max_mem": max_mem}


ALGORITMOS = [("BFS", bfs), ("DFS", dfs), ("UCS", ucs), ("IDS", ids)]


def rodar(mapa, custos, nome_cenario):
    prob = Problema(mapa, custos)
    prob_real = Problema(mapa, CUSTO_REAL)    # custo "verdadeiro" do percurso
    linhas = {}
    for nome, alg in ALGORITMOS:
        t0 = time.perf_counter()
        r = alg(prob)
        dt = time.perf_counter() - t0
        cam = r["caminho"]
        linhas[nome] = {
            "passos": len(cam) - 1,
            "custo_cenario": prob.custo_caminho(cam),
            "custo_real": prob_real.custo_caminho(cam),
            "expandidos": r["expandidos"],
            "max_mem": r["max_mem"],
            "tempo_ms": 1000 * dt,
            "caminho": cam,
        }
        print(f"[{nome_cenario}] {nome}: passos={len(cam)-1} "
              f"custo={linhas[nome]['custo_cenario']} "
              f"custo_real={linhas[nome]['custo_real']} "
              f"exp={r['expandidos']} mem={r['max_mem']} t={1000*dt:.1f} ms")
    return linhas


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]   # BFS, DFS, UCS, IDS


def figura_mapas(mapa, cen_b, arq):
    # 0 obstáculo, 1 piso, 2 grama, 3 lama (cores de terreno, não de série)
    cmap = ListedColormap(["#3a3a38", "#f4f3ef", "#cfe6c2", "#c9a27a"])
    img = np.where(mapa == OBST, 0, mapa + 1)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), dpi=200)
    for ax, (nome, _), cor in zip(axes.ravel(), ALGORITMOS, CORES):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        cam = np.array(cen_b[nome]["caminho"])
        ax.plot(cam[:, 1], cam[:, 0], color=cor, linewidth=1.8)
        ax.plot(0, 0, "o", color=TINTA, markersize=4)
        ax.plot(mapa.shape[1] - 1, mapa.shape[0] - 1, "*", color=TINTA,
                markersize=7)
        r = cen_b[nome]
        ax.set_title(f"{nome}: {r['passos']} passos, custo {r['custo_real']}\n"
                     f"{r['expandidos']} nós expandidos", fontsize=8,
                     color=TINTA)
        ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.patches import Patch
    leg = [Patch(color="#f4f3ef", label="piso (1)"),
           Patch(color="#cfe6c2", label="grama (3)"),
           Patch(color="#c9a27a", label="lama (8)"),
           Patch(color="#3a3a38", label="obstáculo")]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False,
               fontsize=8)
    fig.subplots_adjust(top=0.85, bottom=0.1, left=0.01, right=0.99, wspace=0.06)
    fig.savefig(arq)
    plt.close(fig)


def figura_escala(escala, tamanhos, arq):
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    for cor, (nome, _) in zip(CORES, ALGORITMOS):
        y = [escala[nome][n] for n in tamanhos]
        ax.plot([n * n for n in tamanhos], y, color=cor, linewidth=2,
                marker="o", markersize=5, label=nome)
    ax.set_xlabel("Número de células do mapa (n x n)", color=TINTA)
    ax.set_ylabel("Nós expandidos (média)", color=TINTA)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA2, labelsize=9)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(arq)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    N = 30
    mapa = gerar_mapa(N, seed=7)
    livres = int((mapa != OBST).sum())
    print(f"Mapa {N}x{N}: {livres} células livres")
    cen_a = rodar(mapa, CUSTO_UNIT, "A: custo unitário")
    cen_b = rodar(mapa, CUSTO_REAL, "B: custos 1/3/8")
    figura_mapas(mapa, cen_b, "figs/p2_caminhos.png")

    # Escalabilidade: média de 10 mapas por tamanho (cenário B)
    tamanhos = [10, 20, 30, 40, 60]
    escala = {nome: {} for nome, _ in ALGORITMOS}
    for n in tamanhos:
        acum = {nome: [] for nome, _ in ALGORITMOS}
        for k in range(10):
            m = gerar_mapa(n, seed=100 + k)
            prob = Problema(m, CUSTO_REAL)
            for nome, alg in ALGORITMOS:
                acum[nome].append(alg(prob)["expandidos"])
        for nome in acum:
            escala[nome][n] = float(np.mean(acum[nome]))
        print("escala", n, {k: round(v[n]) for k, v in escala.items()})
    figura_escala(escala, tamanhos, "figs/p2_escala.png")

    for cen in (cen_a, cen_b):
        for r in cen.values():
            r.pop("caminho")
    with open("results/p2.json", "w", encoding="utf-8") as f:
        json.dump({"N": N, "celulas_livres": livres, "cenario_A": cen_a,
                   "cenario_B": cen_b, "escala": escala},
                  f, ensure_ascii=False, indent=1)
