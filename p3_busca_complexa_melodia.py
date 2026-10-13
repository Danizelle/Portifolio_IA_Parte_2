"""
Projeto 3 - Busca em ambientes complexos: compor uma pequena melodia
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
Compor uma melodia de 16 notas (dois compassos de colcheias, 120 bpm)
partindo de notas sorteadas ao acaso. Não há "qualidade artística universal":
a melodia é avaliada por uma PONTUAÇÃO que mede regras musicais simples,
definidas pelo autor. Maximizar a pontuação é um problema de busca local
(o caminho não importa, só a melodia final) num espaço de 25^16 ~ 2 x 10^22
sequências.

Pontuação (quanto maior, melhor)
  R1 escala:    -2 por nota fora de dó maior;
  R2 saltos:    -1 por intervalo maior que uma 4a justa (5 semitons),
                -3 se maior que uma oitava;
  R3 padrão:    +1 por intervalo do 2o compasso igual ao do 1o (o 2o compasso
                repete o desenho melódico do 1o, até 7 pontos);
  R4 tônica:    +3 se começa em dó, +3 se termina em dó;
  R5 variedade: -1 por nota igual à anterior; -2 se há 3 ou menos alturas
                distintas;
  R6 âmbito:    -1 por semitom de extensão acima de uma oitava.
Máximo possível: 13 pontos (R3 = 7 e R4 = 6, sem penalidades).

Métodos (vizinho = trocar uma nota por outra altura ou deslocá-la 1-2 semitons)
  * Subida de encosta (primeiro vizinho que melhora; para após 300 tentativas
    sem melhora);
  * Subida de encosta com reinícios aleatórios (10 reinícios);
  * Têmpera simulada (resfriamento geométrico, T0 = 3, Tf = 0,05).

Saídas: áudio (WAV) da melodia inicial e final, sequência de notas,
evolução da pontuação, estatísticas de 30 execuções por método.

Execução:  python p3_busca_complexa_melodia.py
Saídas:    figs/p3_*.png, results/p3.json, results/p3_melodia_*.wav
"""

import json
import math
import os
import random
import time
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_NOTAS = 16
ALTURA_MIN, ALTURA_MAX = 60, 84                    # dó4 .. dó6 (MIDI), 25 alturas
ESCALA = {0, 2, 4, 5, 7, 9, 11}                    # classes de altura de dó maior
NOMES = ["dó", "dó#", "ré", "ré#", "mi", "fá", "fá#", "sol", "sol#", "lá", "lá#", "si"]
PONTUACAO_MAX = 13


def nome(midi):
    return f"{NOMES[midi % 12]}{midi // 12 - 1}"


# --------------------------------------------------------------------------
# Pontuação
# --------------------------------------------------------------------------
def pontuar(mel, detalhar=False):
    """Soma das regras R1-R6. Com detalhar=True devolve também cada parcela."""
    mel = list(mel)
    inter = [b - a for a, b in zip(mel, mel[1:])]
    r1 = -2 * sum(1 for n in mel if n % 12 not in ESCALA)
    r2 = sum(-3 if abs(i) > 12 else (-1 if abs(i) > 5 else 0) for i in inter)
    meio = N_NOTAS // 2
    r3 = sum(1 for i in range(meio - 1) if inter[i] == inter[meio + i])
    r4 = (3 if mel[0] % 12 == 0 else 0) + (3 if mel[-1] % 12 == 0 else 0)
    r5 = -sum(1 for i in inter if i == 0) - (2 if len(set(mel)) <= 3 else 0)
    r6 = -max(0, (max(mel) - min(mel)) - 12)
    total = r1 + r2 + r3 + r4 + r5 + r6
    if detalhar:
        return total, {"R1 escala": r1, "R2 saltos": r2, "R3 padrão": r3,
                       "R4 tônica": r4, "R5 variedade": r5, "R6 âmbito": r6}
    return total


# --------------------------------------------------------------------------
# Vizinhança e métodos de busca local
# --------------------------------------------------------------------------
def melodia_aleatoria(rng):
    return [rng.randint(ALTURA_MIN, ALTURA_MAX) for _ in range(N_NOTAS)]


def vizinho(mel, rng):
    """Troca uma nota: metade das vezes por uma altura qualquer, metade por um
    deslocamento de 1 ou 2 semitons (busca fina)."""
    novo = list(mel)
    i = rng.randrange(N_NOTAS)
    if rng.random() < 0.5:
        novo[i] = rng.randint(ALTURA_MIN, ALTURA_MAX)
    else:
        novo[i] = min(ALTURA_MAX, max(ALTURA_MIN, novo[i] + rng.choice([-2, -1, 1, 2])))
    return novo


def subida_encosta(mel, rng, paciencia=300, historico=None):
    """Primeira melhora: aceita o primeiro vizinho sorteado que não piora;
    para quando `paciencia` vizinhos seguidos não melhoram (ótimo local)."""
    atual, p = list(mel), pontuar(mel)
    sem_melhora, aval = 0, 1
    if historico is not None:
        historico.append(p)
    while sem_melhora < paciencia:
        cand = vizinho(atual, rng)
        pc = pontuar(cand)
        aval += 1
        if pc > p:
            atual, p, sem_melhora = cand, pc, 0
        else:
            sem_melhora += 1
        if historico is not None:
            historico.append(p)
    return atual, p, aval


def reinicio_aleatorio(mel, rng, reinicios=10, historico=None):
    melhor, pm, total = None, -math.inf, 0
    for k in range(reinicios):
        ini = mel if k == 0 else melodia_aleatoria(rng)
        hist = [] if historico is not None else None
        m, p, aval = subida_encosta(ini, rng, historico=hist)
        total += aval
        if historico is not None:
            historico.extend(max(pm, x) for x in hist)   # melhor global até aqui
        if p > pm:
            melhor, pm = m, p
    return melhor, pm, total


def tempera_simulada(mel, rng, T0=3.0, Tf=0.05, iteracoes=20000, historico=None):
    atual, p = list(mel), pontuar(mel)
    melhor, pm = atual, p
    alfa = (Tf / T0) ** (1.0 / iteracoes)
    T = T0
    if historico is not None:
        historico.append(pm)
    for it in range(iteracoes):
        cand = vizinho(atual, rng)
        pc = pontuar(cand)
        delta = pc - p
        if delta >= 0 or rng.random() < math.exp(delta / T):
            atual, p = cand, pc
            if p > pm:
                melhor, pm = atual, p
        T *= alfa
        if historico is not None:
            historico.append(pm)
    return melhor, pm, iteracoes + 1


# --------------------------------------------------------------------------
# Áudio
# --------------------------------------------------------------------------
def sintetizar(mel, caminho, bpm=120, taxa=44100):
    """Cada nota é uma colcheia (0,25 s a 120 bpm): senoide com dois harmônicos
    e envelope de ataque/decaimento, gravada em WAV mono de 16 bits."""
    dur = 60.0 / bpm / 2
    t = np.arange(int(taxa * dur)) / taxa
    env = np.minimum(1, t / 0.01) * np.exp(-3.0 * t)       # ataque curto, decai
    sinal = []
    for n in mel:
        f = 440.0 * 2 ** ((n - 69) / 12)
        onda = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
                + 0.2 * np.sin(6 * np.pi * f * t))
        sinal.append(onda * env)
    s = np.concatenate(sinal)
    s = (0.8 * s / np.abs(s).max() * 32767).astype(np.int16)
    with wave.open(caminho, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(taxa)
        w.writeframes(s.tobytes())


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA2, labelsize=8.5)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_piano_roll(inicial, final, p_ini, p_fim):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), dpi=200, sharey=True)
    for ax, mel, titulo, cor in ((axes[0], inicial, f"(a) inicial: {p_ini} pontos", LARANJA),
                                 (axes[1], final, f"(b) final (têmpera simulada): {p_fim} pontos", AZUL)):
        for alt in range(ALTURA_MIN, ALTURA_MAX + 1):          # fundo: teclas da escala
            if alt % 12 in ESCALA:
                ax.axhspan(alt - 0.5, alt + 0.5, color="#f4f3ef", lw=0)
        ax.axvline(7.5, color=TINTA2, lw=0.8, ls="--")
        for i, n in enumerate(mel):
            fora = n % 12 not in ESCALA
            ax.add_patch(plt.Rectangle((i - 0.45, n - 0.45), 0.9, 0.9, color=cor,
                                       alpha=0.5 if fora else 1, lw=0))
        ax.plot(range(N_NOTAS), mel, color=cor, lw=0.8, alpha=0.5)
        ax.set_xlim(-0.6, N_NOTAS - 0.4)
        ax.set_ylim(ALTURA_MIN - 0.6, ALTURA_MAX + 0.6)
        ax.set_xticks(range(N_NOTAS)); ax.set_xticklabels(range(1, N_NOTAS + 1))
        ax.set_yticks([a for a in range(ALTURA_MIN, ALTURA_MAX + 1) if a % 12 == 0])
        ax.set_yticklabels([nome(a) for a in range(ALTURA_MIN, ALTURA_MAX + 1) if a % 12 == 0])
        ax.set_xlabel("Nota (colcheia); compassos 1 | 2", color=TINTA, fontsize=9)
        ax.set_title(titulo, fontsize=9, color=TINTA2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TINTA2, labelsize=8)
    axes[0].set_ylabel("Altura (MIDI)", color=TINTA, fontsize=9)
    fig.text(0.5, -0.03, "faixas claras = notas da escala de dó maior; quadrados pálidos = notas fora da escala",
             ha="center", fontsize=8, color=TINTA2)
    fig.tight_layout()
    fig.savefig("figs/p3_piano_roll.png", bbox_inches="tight")
    plt.close(fig)


def figura_evolucao(hist, pm):
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    for chave, cor, rot, lw in (("reinicio", AQUA, "reinício aleatório (10)", 2.6),
                                ("subida", LARANJA, "subida de encosta", 1.3),
                                ("tempera", AZUL, "têmpera simulada", 1.4)):
        h = hist[chave]
        ax.plot(np.arange(1, len(h) + 1), h, color=cor, lw=lw, label=rot)
    ax.annotate("ótimo local da\nsubida de encosta", xy=(len(hist["subida"]), hist["subida"][-1]),
                xytext=(len(hist["subida"]) * 1.6, hist["subida"][-1] - 10), fontsize=7.5, color=LARANJA,
                arrowprops=dict(arrowstyle="-", color=LARANJA, lw=0.8))
    ax.axhline(pm, color=TINTA2, ls=":", lw=1)
    ax.text(1.2, pm + 0.4, f"máximo = {pm}", fontsize=8, color=TINTA2)
    ax.set_xscale("log")
    ax.set_xlabel("Melodias avaliadas", color=TINTA, fontsize=9)
    ax.set_ylabel("Melhor pontuação", color=TINTA, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    estilo(ax)
    fig.savefig("figs/p3_evolucao.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------
METODOS = {"Subida de encosta": subida_encosta,
           "Reinício aleatório (10)": reinicio_aleatorio,
           "Têmpera simulada": tempera_simulada}

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    EXEC = 30
    res = {m: {"pont": [], "aval": [], "t": []} for m in METODOS}
    hist1, exemplo = {}, {}
    for e in range(EXEC):
        rng = random.Random(e)
        inicial = melodia_aleatoria(rng)             # mesma melodia inicial para os 3 métodos
        for nome_m, fn in METODOS.items():
            rng_m = random.Random(1000 + e)
            h = [] if e == 0 else None
            t0 = time.perf_counter()
            mel, p, aval = fn(inicial, rng_m, historico=h)
            dt = time.perf_counter() - t0
            res[nome_m]["pont"].append(p); res[nome_m]["aval"].append(aval); res[nome_m]["t"].append(dt)
            assert pontuar(mel) == p
            if e == 0:
                chave = {"Subida de encosta": "subida", "Reinício aleatório (10)": "reinicio",
                         "Têmpera simulada": "tempera"}[nome_m]
                hist1[chave] = h
                exemplo[chave] = (mel, p)
        if e == 0:
            exemplo["inicial"] = (inicial, pontuar(inicial))

    print(f"{'método':26s} {'média':>6s} {'dp':>5s} {'min':>4s} {'max':>4s} {'máx atingido':>13s} {'avaliações':>11s} {'tempo':>8s}")
    resumo = {}
    for m, r in res.items():
        p = np.array(r["pont"])
        resumo[m] = {"media": float(p.mean()), "dp": float(p.std()), "min": int(p.min()), "max": int(p.max()),
                     "atingiu_max": int((p == PONTUACAO_MAX).sum()), "aval_medio": float(np.mean(r["aval"])),
                     "t_medio_s": float(np.mean(r["t"]))}
        print(f"{m:26s} {p.mean():6.2f} {p.std():5.2f} {p.min():4d} {p.max():4d} "
              f"{resumo[m]['atingiu_max']:10d}/30 {np.mean(r['aval']):11.0f} {np.mean(r['t'])*1000:6.0f} ms")
    p_ini = np.array([pontuar(melodia_aleatoria(random.Random(e))) for e in range(EXEC)])
    print(f"melodias iniciais: média {p_ini.mean():.2f}, dp {p_ini.std():.2f}")

    # ---- exemplo (execução 1): áudio, notas e parcelas da pontuação ----
    ini, p_i = exemplo["inicial"]
    fim, p_f = exemplo["tempera"]
    sintetizar(ini, "results/p3_melodia_inicial.wav")
    sintetizar(fim, "results/p3_melodia_final.wav")
    _, det_i = pontuar(ini, True)
    _, det_f = pontuar(fim, True)
    print("\ninicial:", " ".join(nome(n) for n in ini), f"= {p_i}", det_i)
    print("final:  ", " ".join(nome(n) for n in fim), f"= {p_f}", det_f)
    print("subida: ", " ".join(nome(n) for n in exemplo["subida"][0]), f"= {exemplo['subida'][1]}",
          pontuar(exemplo["subida"][0], True)[1])
    figura_piano_roll(ini, fim, p_i, p_f)
    figura_evolucao(hist1, PONTUACAO_MAX)

    with open("results/p3.json", "w") as f:
        json.dump({"n_notas": N_NOTAS, "alturas": [ALTURA_MIN, ALTURA_MAX], "pontuacao_max": PONTUACAO_MAX,
                   "execucoes": EXEC, "iniciais": {"media": float(p_ini.mean()), "dp": float(p_ini.std())},
                   "resumo": resumo,
                   "exemplo": {"inicial": {"midi": ini, "notas": [nome(n) for n in ini], "pontos": p_i, "parcelas": det_i},
                               "final_tempera": {"midi": fim, "notas": [nome(n) for n in fim], "pontos": p_f, "parcelas": det_f},
                               "final_subida": {"midi": exemplo["subida"][0], "notas": [nome(n) for n in exemplo["subida"][0]],
                                                "pontos": exemplo["subida"][1],
                                                "parcelas": pontuar(exemplo["subida"][0], True)[1]}}},
                  f, indent=1, ensure_ascii=False)
    print("ok: figs/p3_*.png, results/p3.json, results/p3_melodia_*.wav")
