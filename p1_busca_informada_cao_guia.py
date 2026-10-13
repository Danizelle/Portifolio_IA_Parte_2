"""
Projeto 1 - Busca informada: cão-guia virtual (A*) em um prédio
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
Um aplicativo de orientação para pessoas cegas precisa planejar uma rota
dentro de um prédio e transformá-la em instruções faladas, do tipo
"siga em frente 6 m pelo piso tátil e vire à direita". O prédio é um mapa em
grade (células de 0,5 m) com paredes, obstáculos (balcão, bancos, vasos),
faixas de PISO TÁTIL direcional e PONTOS DE ORIENTAÇÃO (entrada, recepção,
elevador, escada, banheiro, salas).

A rota boa não é só a mais curta: deve preferir o piso tátil, manter distância
de obstáculos e ter poucas curvas (cada curva é uma instrução a mais e um
risco de desorientação). Isso entra no custo de passo:
  * entrar numa célula comum custa 1,0; numa célula de piso tátil, 0,7;
  * célula encostada em parede/obstáculo: +0,5 (margem de segurança);
  * virar 90 graus custa 0,3 (o estado inclui a direção em que se olha).

Algoritmos (mesma busca de melhor escolha em grafo, mudando f):
  * Custo uniforme (UCS): f = g            -> ótimo, sem informação;
  * A*:                   f = g + h        -> ótimo (h admissível);
  * A* ponderado (w = 2): f = g + 2h       -> mais rápido, sem garantia;
  * Gulosa:               f = h.
Heurística: h = 0,7 * distância de Manhattan (0,7 é o menor custo de passo,
logo h nunca superestima e é consistente).

Saída: para 30 rotas (todos os pares ordenados de 6 pontos de orientação)
compara-se nós expandidos, custo, comprimento, curvas e uso do piso tátil;
e para uma rota de exemplo imprimem-se as instruções geradas.

Execução:  python p1_busca_informada_cao_guia.py
Saídas:    figs/p1_*.png  e  results/p1.json
"""

import heapq
import itertools
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LIVRE, PAREDE, OBST, TATIL = 0, 1, 2, 3
CELULA_M = 0.5                                     # metros por célula
DIRS = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}   # leste, sul, oeste, norte
NOME_DIR = {0: "leste", 1: "sul", 2: "oeste", 3: "norte"}


# --------------------------------------------------------------------------
# O prédio
# --------------------------------------------------------------------------
def construir_predio(seed=2026):
    """Andar de 20 m x 30 m em forma de cruz: corredor horizontal (leste-oeste),
    corredor vertical (norte-sul) e um saguão no cruzamento, com balcão da
    recepção e duas colunas. Salas nos quatro quadrantes, elevador no fim do
    corredor norte e escada no fim do corredor sul. Piso tátil direcional nos
    dois corredores, com ramais para a recepção, o elevador e a escada."""
    L, C = 40, 60
    m = np.full((L, C), PAREDE, dtype=np.int8)
    m[18:22, 1:59] = LIVRE                           # corredor horizontal (2 m)
    m[1:39, 28:32] = LIVRE                           # corredor vertical (2 m)
    m[14:26, 23:37] = LIVRE                          # saguão (6 m x 7 m)
    m[18:22, 0] = LIVRE                              # entrada: abertura na parede oeste
    # portas (célula de parede aberta) e célula de referência em frente a cada uma
    portas = {"sala 101": (17, 8), "sala 102": (17, 44), "sala 103": (22, 8),
              "sala 104": (22, 44), "sala 105": (10, 27), "sala 106": (30, 32),
              "banheiro": (22, 52), "elevador": (0, 29), "escada": (39, 29)}
    for l, c in portas.values():
        m[l, c:c + 2] = LIVRE
    pontos = {"entrada": (19, 1), "recepção": (20, 26), "sala 101": (18, 8),
              "banheiro": (21, 52), "elevador": (1, 29), "escada": (38, 29)}
    # obstáculos fixos: balcão da recepção, duas colunas, bancos
    m[14:17, 24:27] = OBST                           # balcão (lado norte do saguão)
    m[16, 34] = m[23, 34] = OBST                     # colunas
    m[18, 14:17] = OBST                              # banco
    m[21, 38:41] = OBST                              # banco
    m[7, 31] = OBST                                  # vaso
    rng = np.random.default_rng(seed)                # 6 obstáculos "do dia": carrinhos, caixas
    cand = [(l, c) for l in (18, 21) for c in range(3, 57) if m[l, c] == LIVRE and not 23 <= c <= 36]
    cand += [(l, c) for c in (28, 31) for l in range(3, 37) if m[l, c] == LIVRE and not 14 <= l <= 25]
    for i in rng.choice(len(cand), 6, replace=False):
        m[cand[i]] = OBST
    # piso tátil: eixo dos dois corredores + ramais
    m[20, 1:58] = TATIL                              # corredor horizontal
    m[1:38, 30] = TATIL                              # corredor vertical
    m[20, 26] = TATIL; m[19, 26] = TATIL; m[18, 26] = TATIL   # ramal até o balcão
    m[1, 29] = TATIL; m[38, 29] = TATIL              # frente do elevador e da escada
    m[m == TATIL] = TATIL
    return m, pontos, portas


# --------------------------------------------------------------------------
# Problema de busca: estado = (linha, coluna, direção)
# --------------------------------------------------------------------------
class Navegacao:
    CUSTO_TATIL, CUSTO_COMUM, MARGEM, CURVA = 0.7, 1.0, 0.5, 0.3

    def __init__(self, mapa, inicio, objetivo, direcao0=0, preferir_tatil=True):
        self.m = mapa
        self.inicio = (inicio[0], inicio[1], direcao0)
        self.objetivo = objetivo
        self.preferir_tatil = preferir_tatil
        # pré-calcula o custo de entrar em cada célula
        L, C = mapa.shape
        bloq = (mapa == PAREDE) | (mapa == OBST)
        perto = np.zeros_like(bloq)
        perto[1:, :] |= bloq[:-1, :]; perto[:-1, :] |= bloq[1:, :]
        perto[:, 1:] |= bloq[:, :-1]; perto[:, :-1] |= bloq[:, 1:]
        base = np.where((mapa == TATIL) & preferir_tatil, self.CUSTO_TATIL, self.CUSTO_COMUM)
        self.custo_celula = base + self.MARGEM * perto
        self.custo_celula[bloq] = np.inf

    def objetivo_atingido(self, s):
        return (s[0], s[1]) == self.objetivo

    def sucessores(self, s):
        l, c, d = s
        dl, dc = DIRS[d]
        nl, nc = l + dl, c + dc
        if 0 <= nl < self.m.shape[0] and 0 <= nc < self.m.shape[1] and np.isfinite(self.custo_celula[nl, nc]):
            yield (nl, nc, d), self.custo_celula[nl, nc], "frente"
        yield (l, c, (d + 1) % 4), self.CURVA, "direita"
        yield (l, c, (d - 1) % 4), self.CURVA, "esquerda"

    def h(self, s):
        # 0,7 * Manhattan: menor custo possível por célula -> admissível e consistente
        return self.CUSTO_TATIL * (abs(s[0] - self.objetivo[0]) + abs(s[1] - self.objetivo[1]))


def busca_melhor_escolha(prob, peso_g=1.0, peso_h=1.0):
    """Busca de melhor escolha em grafo com f = peso_g*g + peso_h*h.
    (UCS: peso_h=0; A*: 1,1; A* ponderado: 1,w; gulosa: peso_g=0.)"""
    s0 = prob.inicio
    g = {s0: 0.0}
    pai = {s0: None}
    fila = [(peso_h * prob.h(s0), 0, s0)]
    fechados = set()
    expandidos = 0
    tie = 1
    while fila:
        f, _, s = heapq.heappop(fila)
        if s in fechados:
            continue
        fechados.add(s)
        expandidos += 1
        if prob.objetivo_atingido(s):
            caminho = []
            while s is not None:
                caminho.append(s)
                s = pai[s]
            return caminho[::-1], g[caminho[0]], expandidos
        for s2, custo, _ in prob.sucessores(s):
            g2 = g[s] + custo
            if g2 < g.get(s2, np.inf):
                g[s2], pai[s2] = g2, s
                heapq.heappush(fila, (peso_g * g2 + peso_h * prob.h(s2), tie, s2))
                tie += 1
    return None, np.inf, expandidos


# --------------------------------------------------------------------------
# Geração de instruções
# --------------------------------------------------------------------------
def lado(d_olhar, d_para):
    """'direita'/'esquerda'/'à frente' de quem olha em d_olhar para algo em d_para."""
    diff = (d_para - d_olhar) % 4
    return {0: "à frente", 1: "à sua direita", 2: "atrás", 3: "à sua esquerda"}[diff]


def pontos_vizinhos(celula, pontos, portas, mapa):
    """Pontos de orientação adjacentes (porta ou célula de referência) à célula."""
    l, c = celula
    achados = []
    for nome, (pl, pc) in list(portas.items()) + [(n, p) for n, p in pontos.items()
                                                   if n not in portas]:
        if max(abs(pl - l), abs(pc - c)) <= 1 and nome != "entrada":
            achados.append((nome, (pl, pc)))
    return achados


def lado_de(celula, ref, d):
    """Lado em que a referência ref está, vista da célula por quem olha em d."""
    dl, dc = ref[0] - celula[0], ref[1] - celula[1]
    d_ref = (1 if dl > 0 else 3) if abs(dl) >= abs(dc) else (0 if dc > 0 else 2)
    return lado(d, d_ref)


def referencias_ao_longo(celulas, portas, d, fim):
    """Portas a até 3 células (1,5 m) de lado do trecho, exceto no seu fim."""
    out = []
    for nome, (pl, pc) in portas.items():
        if max(abs(pl - fim[0]), abs(pc - fim[1])) <= 1:
            continue                                   # já citada como fim do trecho
        # célula do trecho mais próxima da porta: define de que lado ela fica
        dist = [abs(pl - l) + abs(pc - c) for l, c in celulas[:-1]]
        if dist and min(dist) <= 3:
            l, c = celulas[int(np.argmin(dist))]
            out.append((nome, lado_de((l, c), (pl, pc), d)))
    return out[:3]


def gerar_instrucoes(caminho, mapa, pontos, portas, destino):
    """Converte o caminho de estados em frases. Agrupa passos 'frente' em
    trechos retos; descreve cada trecho por distância, piso tátil e
    ponto de orientação no fim; curvas viram 'vire à direita/esquerda'."""
    frases = []
    i = 0
    n = len(caminho)
    while i < n - 1:
        s, s2 = caminho[i], caminho[i + 1]
        if s2[2] != s[2]:                                   # curva
            giro = "direita" if (s2[2] - s[2]) % 4 == 1 else "esquerda"
            # curvas consecutivas (meia-volta)
            if i + 2 < n and caminho[i + 2][2] != s2[2] and (caminho[i + 2][2] - s2[2]) % 4 == (s2[2] - s[2]) % 4:
                frases.append("faça meia-volta")
                i += 2
            else:
                frases.append(f"vire à {giro}")
                i += 1
            continue
        # trecho reto
        j = i
        while j + 1 < n and caminho[j + 1][2] == s[2] and (caminho[j + 1][0], caminho[j + 1][1]) != (caminho[j][0], caminho[j][1]):
            j += 1
        celulas = [(caminho[k][0], caminho[k][1]) for k in range(i + 1, j + 1)]
        dist = len(celulas) * CELULA_M
        tatil = [mapa[l, c] == TATIL for l, c in celulas]
        frac_tatil = np.mean(tatil) if tatil else 0
        fim = celulas[-1]
        d = s[2]
        txt = f"siga em frente {dist:g} m".replace(".", ",")
        if frac_tatil >= 0.8:
            txt += " pelo piso tátil"
        elif tatil and tatil[-1] and not tatil[0]:
            txt += " até o piso tátil"
        elif tatil and tatil[0] and not tatil[-1] and frac_tatil >= 0.5:
            txt += " até o fim do piso tátil"
        if fim == destino:
            txt += f"; você chegou: {[k for k, v in pontos.items() if v == destino][0]}"
        else:
            refs = pontos_vizinhos(fim, pontos, portas, mapa)
            if refs:
                nome, (pl, pc) = refs[0]
                txt += f", até {nome} ({lado_de(fim, (pl, pc), d)})"
        # referências que ficam ao lado do trecho (portas a até 1 m do caminho)
        passadas = referencias_ao_longo(celulas, portas, d, fim)
        if passadas:
            txt += "; você passa por " + ", ".join(f"{n} ({l})" for n, l in passadas)
        frases.append(txt)
        i = j
    return frases


# --------------------------------------------------------------------------
# Métricas de um caminho
# --------------------------------------------------------------------------
def metricas(caminho, mapa):
    passos = [(a, b) for a, b in zip(caminho, caminho[1:]) if (a[0], a[1]) != (b[0], b[1])]
    curvas = sum(1 for a, b in zip(caminho, caminho[1:]) if a[2] != b[2])
    tatil = sum(1 for _, b in passos if mapa[b[0], b[1]] == TATIL)
    return {"comprimento_m": len(passos) * CELULA_M, "curvas": curvas,
            "frac_tatil": tatil / max(1, len(passos))}


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
AZUL, LARANJA, AQUA, AMARELO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA2, labelsize=8.5)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def desenhar_mapa(ax, mapa, pontos, portas):
    cores = {LIVRE: "#ffffff", PAREDE: "#dedbd4", OBST: "#8a877f", TATIL: "#f3d27a"}
    rgb = np.zeros(mapa.shape + (3,))
    for v, hexc in cores.items():
        rgb[mapa == v] = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    ax.imshow(rgb, interpolation="nearest")
    for nome, (l, c) in portas.items():
        ax.add_patch(Rectangle((c - 0.5, l - 0.5), 2, 1, fill=False, ec=TINTA2, lw=0.8, ls=":"))
        if l in (0, 39):                     # elevador / escada: rótulo ao lado
            ax.text(c + 3.2, l + (1.5 if l == 0 else -1.5), nome, fontsize=5.5, va="center", color=TINTA2)
        elif c in (27, 32):                  # portas nas paredes do corredor vertical
            ax.text(c + (-1.5 if c == 27 else 3.5), l + 0.5, nome, fontsize=5.5, va="center",
                    ha="right" if c == 27 else "left", color=TINTA2)
        else:
            ax.text(c + 1, l + (2.2 if l > 20 else -1.2), nome, fontsize=5.5, ha="center",
                    va="center", color=TINTA2)
    for nome, (l, c) in pontos.items():
        ax.plot(c, l, "o", ms=3.5, color=TINTA, mfc="white", mew=0.9)
    ax.text(pontos["entrada"][1] + 1, pontos["entrada"][0] - 2.2, "entrada", fontsize=5.5, color=TINTA2)
    ax.text(25.5, 13, "recepção", fontsize=5.5, color=TINTA2, ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def figura_rotas(mapa, pontos, portas, rotas):
    """rotas: lista de (rótulo, cor, caminho)."""
    fig, ax = plt.subplots(figsize=(9, 6.3), dpi=200)
    desenhar_mapa(ax, mapa, pontos, portas)
    estilos = [dict(lw=3.0), dict(lw=1.6), dict(lw=1.6, ls=(0, (3, 2)))]
    for (rot, cor, cam), est in zip(rotas, estilos):
        ys = [s[0] for s in cam]; xs = [s[1] for s in cam]
        ax.plot(xs, ys, color=cor, alpha=0.95, label=rot, solid_capstyle="round", **est)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    fig.savefig("figs/p1_rotas.png", bbox_inches="tight")
    plt.close(fig)


def figura_esforco(res):
    algs = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=200)
    cores = {"UCS": TINTA2, "A*": AZUL, "A* ponderado (w=2)": AQUA, "Gulosa": LARANJA}
    ax = axes[0]
    for a in algs:
        ax.scatter(res[a]["h_inicial"], res[a]["expandidos"], s=14, color=cores[a], label=a, alpha=0.85)
    ax.set_xlabel("Estimativa inicial h(início) (custo)", color=TINTA, fontsize=9)
    ax.set_ylabel("Nós expandidos", color=TINTA, fontsize=9)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) esforço em cada uma das 30 rotas", fontsize=9, color=TINTA2)
    estilo(ax)
    ax = axes[1]
    for a in algs:
        exc = 100 * (np.array(res[a]["custo"]) / np.array(res["UCS"]["custo"]) - 1)
        ax.scatter(np.full(len(exc), algs.index(a)) + np.linspace(-0.15, 0.15, len(exc)), exc,
                   s=14, color=cores[a], alpha=0.85)
    ax.set_xticks(range(len(algs))); ax.set_xticklabels(algs, fontsize=8)
    ax.set_ylabel("Custo acima do ótimo (%)", color=TINTA, fontsize=9)
    ax.set_title("(b) qualidade da rota", fontsize=9, color=TINTA2)
    estilo(ax)
    fig.tight_layout()
    fig.savefig("figs/p1_esforco.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------
ALGS = {"UCS": (1, 0), "A*": (1, 1), "A* ponderado (w=2)": (1, 2), "Gulosa": (0, 1)}

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    mapa, pontos, portas = construir_predio()
    print(f"prédio {mapa.shape[0]}x{mapa.shape[1]} células de {CELULA_M} m; "
          f"{(mapa == LIVRE).sum() + (mapa == TATIL).sum()} células livres, "
          f"{(mapa == TATIL).sum()} de piso tátil, {(mapa == OBST).sum()} de obstáculo")

    # ---- 30 rotas: todos os pares ordenados dos 6 pontos ----
    res = {a: {"expandidos": [], "custo": [], "t_ms": [], "comprimento_m": [], "curvas": [],
               "frac_tatil": [], "h_inicial": []} for a in ALGS}
    pares = list(itertools.permutations(pontos, 2))
    for o, d in pares:
        prob = Navegacao(mapa, pontos[o], pontos[d])
        for a, (pg, ph) in ALGS.items():
            t0 = time.perf_counter()
            cam, custo, exp = busca_melhor_escolha(prob, pg, ph)
            dt = (time.perf_counter() - t0) * 1000
            assert cam is not None
            m = metricas(cam, mapa)
            r = res[a]
            r["expandidos"].append(exp); r["custo"].append(custo); r["t_ms"].append(dt)
            r["comprimento_m"].append(m["comprimento_m"]); r["curvas"].append(m["curvas"])
            r["frac_tatil"].append(m["frac_tatil"]); r["h_inicial"].append(prob.h(prob.inicio))
    # A* deve ser sempre ótimo (mesmo custo do UCS)
    assert np.allclose(res["A*"]["custo"], res["UCS"]["custo"])
    resumo = {}
    for a in ALGS:
        r = res[a]
        exc = 100 * (np.array(r["custo"]) / np.array(res["UCS"]["custo"]) - 1)
        resumo[a] = {"expandidos_medio": float(np.mean(r["expandidos"])),
                     "expandidos_max": int(np.max(r["expandidos"])),
                     "t_medio_ms": float(np.mean(r["t_ms"])),
                     "custo_medio": float(np.mean(r["custo"])),
                     "excesso_medio_pct": float(exc.mean()), "excesso_max_pct": float(exc.max()),
                     "rotas_otimas": int((exc < 1e-9).sum()),
                     "comprimento_medio_m": float(np.mean(r["comprimento_m"])),
                     "curvas_medias": float(np.mean(r["curvas"])),
                     "frac_tatil_media": float(np.mean(r["frac_tatil"]))}
        print(f"{a:20s} exp {resumo[a]['expandidos_medio']:7.0f}  t {resumo[a]['t_medio_ms']:5.1f} ms  "
              f"custo {resumo[a]['custo_medio']:6.2f}  excesso {resumo[a]['excesso_medio_pct']:5.2f}% "
              f"(max {resumo[a]['excesso_max_pct']:5.2f}%)  ótimas {resumo[a]['rotas_otimas']}/30  "
              f"compr {resumo[a]['comprimento_medio_m']:5.1f} m  curvas {resumo[a]['curvas_medias']:.1f}  "
              f"tátil {100*resumo[a]['frac_tatil_media']:.0f}%")

    # ---- efeito de preferir o piso tátil (A*, mesmas 30 rotas) ----
    sem = {"comprimento_m": [], "curvas": [], "frac_tatil": [], "expandidos": []}
    for o, d in pares:
        prob = Navegacao(mapa, pontos[o], pontos[d], preferir_tatil=False)
        cam, custo, exp = busca_melhor_escolha(prob, 1, 1)
        m = metricas(cam, mapa)
        for k in ("comprimento_m", "curvas", "frac_tatil"):
            sem[k].append(m[k])
        sem["expandidos"].append(exp)
    resumo["A* sem preferência pelo piso tátil"] = {k: float(np.mean(v)) for k, v in sem.items()}
    print("A* sem piso tátil: compr %.1f m  curvas %.1f  tátil %.0f%%  exp %.0f" % (
        np.mean(sem["comprimento_m"]), np.mean(sem["curvas"]), 100 * np.mean(sem["frac_tatil"]),
        np.mean(sem["expandidos"])))

    # ---- rota de exemplo: entrada -> escada ----
    o, d = "entrada", "elevador"
    prob = Navegacao(mapa, pontos[o], pontos[d])
    cam_astar, custo, exp = busca_melhor_escolha(prob, 1, 1)
    cam_gul, custo_g, exp_g = busca_melhor_escolha(prob, 0, 1)
    prob2 = Navegacao(mapa, pontos[o], pontos[d], preferir_tatil=False)
    cam_sem, custo_s, _ = busca_melhor_escolha(prob2, 1, 1)
    instr = gerar_instrucoes(cam_astar, mapa, pontos, portas, pontos[d])
    print(f"\nRota {o} -> {d}: custo {custo:.1f}, {exp} nós expandidos, "
          f"{metricas(cam_astar, mapa)}")
    for k, f in enumerate(instr, 1):
        print(f"  {k}. {f}")
    instr_gul = gerar_instrucoes(cam_gul, mapa, pontos, portas, pontos[d])
    print(f"Gulosa: custo {custo_g:.1f}, {exp_g} nós, {len(instr_gul)} instruções, {metricas(cam_gul, mapa)}")
    figura_rotas(mapa, pontos, portas,
                 [("A* (prefere piso tátil)", AZUL, cam_astar),
                  ("gulosa", LARANJA, cam_gul),
                  ("A* sem preferência pelo piso tátil", AQUA, cam_sem)])
    figura_esforco(res)

    with open("results/p1.json", "w") as f:
        json.dump({"mapa": {"linhas": int(mapa.shape[0]), "colunas": int(mapa.shape[1]),
                            "celula_m": CELULA_M, "livres": int(((mapa == LIVRE) | (mapa == TATIL)).sum()),
                            "tatil": int((mapa == TATIL).sum()), "obstaculos": int((mapa == OBST).sum())},
                   "custos": {"tatil": Navegacao.CUSTO_TATIL, "comum": Navegacao.CUSTO_COMUM,
                              "margem": Navegacao.MARGEM, "curva": Navegacao.CURVA},
                   "n_rotas": len(pares), "resumo": resumo,
                   "exemplo": {"rota": f"{o} -> {d}", "custo": custo, "expandidos": exp,
                               "metricas": metricas(cam_astar, mapa), "instrucoes": instr,
                               "gulosa": {"custo": custo_g, "expandidos": exp_g,
                                          "metricas": metricas(cam_gul, mapa), "instrucoes": instr_gul},
                               "sem_tatil": {"custo": custo_s, "metricas": metricas(cam_sem, mapa)}}},
                  f, indent=1, ensure_ascii=False)
    print("ok: figs/p1_*.png, results/p1.json")
