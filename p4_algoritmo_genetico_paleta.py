"""
Projeto 4 - Algoritmo genético: paleta de cores de uma fotografia
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
Dada uma fotografia, encontrar uma paleta de k cores que a represente bem e
gerar uma versão estilizada (cada pixel recebe a cor da paleta mais próxima).
Quanto menor k, mais "cartaz" fica a imagem e maior o erro; o experimento
mede esse compromisso para k em {2, 4, 8, 16}.

Formulação como otimização
  * Indivíduo: paleta P = (c_1, ..., c_k), cada c_j um ponto RGB em [0,1]^3
    (3k genes reais);
  * Aptidão: o negativo do erro quadrático médio (MSE) da quantização,
        E(P) = (1/n) * sum_pixels min_j || x_i - c_j ||^2,
    calculado sobre uma amostra fixa de pixels da foto (o erro final é
    medido sobre todos os pixels).

É o problema de quantização de cores. A referência é o k-means (Lloyd, 1982),
que converge para um mínimo local do mesmo erro, e a quantização uniforme
(paleta fixa, sem olhar para a foto).

Algoritmo genético (AG)
  * Seleção por torneio (3);
  * Cruzamento aritmético (cada cor do filho é uma mistura convexa das cores
    dos pais, com peso sorteado), taxa pc;
  * Mutação gaussiana (perturba cada canal com prob. pm, desvio sigma);
  * Elitismo: os 2 melhores sobrevivem;
  * Variante memética: a cada geração a elite recebe um passo de Lloyd.

Imagem: "grace_hopper.jpg", fotografia distribuída com o Matplotlib
(domínio público, U.S. Navy), 600 x 512 pixels.

Execução:  python p4_algoritmo_genetico_paleta.py
Saídas:    figs/p4_*.png  e  results/p4.json
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cbook


# --------------------------------------------------------------------------
# Imagem e erro de quantização
# --------------------------------------------------------------------------
def carregar_foto():
    caminho = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    img = mpimg.imread(caminho).astype(np.float64) / 255.0     # (H, W, 3)
    return img


def quantizar(pixels, paleta):
    """Índice da cor da paleta mais próxima de cada pixel (distância RGB)."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; o termo ||x||^2 não muda o argmin
    d = -2.0 * pixels @ paleta.T + (paleta ** 2).sum(1)[None, :]
    return d.argmin(1)


def erro(pixels, paleta):
    """MSE da quantização (média sobre pixels, soma sobre os 3 canais)."""
    idx = quantizar(pixels, paleta)
    dif = pixels - paleta[idx]
    return float((dif ** 2).sum(1).mean())


def rmse255(mse):
    """Converte o MSE (soma dos 3 canais, escala 0-1) em RMSE por canal 0-255."""
    return 255.0 * np.sqrt(mse / 3.0)


# --------------------------------------------------------------------------
# Referências: k-means (Lloyd) e paleta uniforme
# --------------------------------------------------------------------------
def passo_lloyd(pixels, paleta, rng):
    """Um passo do k-means: cada cor vira a média dos pixels que a escolheram."""
    idx = quantizar(pixels, paleta)
    nova = paleta.copy()
    for j in range(len(paleta)):
        sel = idx == j
        if sel.any():
            nova[j] = pixels[sel].mean(0)
        else:                                   # cor "vazia": reposiciona num pixel
            nova[j] = pixels[rng.integers(len(pixels))]
    return nova


def kmeans(pixels, k, rng, iteracoes=30):
    paleta = pixels[rng.choice(len(pixels), k, replace=False)]
    for _ in range(iteracoes):
        nova = passo_lloyd(pixels, paleta, rng)
        if np.allclose(nova, paleta):
            break
        paleta = nova
    return paleta


def paleta_uniforme(k):
    """Paleta fixa: níveis igualmente espaçados; para k=2 preto e branco,
    k=4 e 16 em cinza, k=8 nos vértices do cubo RGB."""
    if k == 8:
        return np.array([[r, g, b] for r in (0, 1) for g in (0, 1) for b in (0, 1)], float)
    niveis = np.linspace(0, 1, k)
    return np.stack([niveis] * 3, 1)


# --------------------------------------------------------------------------
# Algoritmo genético
# --------------------------------------------------------------------------
class AGPaleta:
    def __init__(self, pixels, k, rng, pop=60, geracoes=120, pc=0.9,
                 pm=None, sigma=0.05, elite=2, memetico=False):
        self.px, self.k, self.rng = pixels, k, rng
        self.pop_n, self.geracoes, self.pc = pop, geracoes, pc
        self.pm = (1.0 / (3 * k)) if pm is None else pm
        self.sigma, self.elite, self.memetico = sigma, elite, memetico

    def aptidao(self, pop):
        return np.array([-erro(self.px, ind) for ind in pop])

    def selecionar(self, apt):
        cand = self.rng.integers(self.pop_n, size=(self.pop_n, 3))   # torneio de 3
        return cand[np.arange(self.pop_n), apt[cand].argmax(1)]

    def cruzar(self, p1, p2):
        if self.rng.random() > self.pc:
            return p1.copy()
        alfa = self.rng.random((self.k, 1))          # um peso por cor
        return alfa * p1 + (1 - alfa) * p2

    def mutar(self, ind):
        mascara = self.rng.random(ind.shape) < self.pm
        ind = ind + mascara * self.rng.normal(0, self.sigma, ind.shape)
        return np.clip(ind, 0, 1)

    def executar(self):
        # população inicial: cores sorteadas entre os pixels da foto
        pop = np.stack([self.px[self.rng.choice(len(self.px), self.k, replace=False)]
                        for _ in range(self.pop_n)])
        apt = self.aptidao(pop)
        hist_melhor, hist_media = [], []
        for g in range(self.geracoes):
            ordem = np.argsort(-apt)
            elite = [pop[i].copy() for i in ordem[:self.elite]]
            if self.memetico:
                elite = [passo_lloyd(self.px, e, self.rng) for e in elite]
            pais = self.selecionar(apt)
            filhos = []
            for i in range(0, self.pop_n - self.elite):
                f = self.cruzar(pop[pais[i]], pop[pais[(i + 1) % self.pop_n]])
                filhos.append(self.mutar(f))
            pop = np.stack(elite + filhos)
            apt = self.aptidao(pop)
            hist_melhor.append(float(-apt.max()))
            hist_media.append(float(-apt.mean()))
        return pop[apt.argmax()], hist_melhor, hist_media


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


def figura_estilizadas(img, paletas, erros):
    """Original e versões estilizadas (AG) com a paleta de cada uma."""
    ks = sorted(paletas)
    fig = plt.figure(figsize=(10, 2.75), dpi=200)
    gs = fig.add_gridspec(2, len(ks) + 1, height_ratios=[12, 1], hspace=0.02, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.set_title("original", fontsize=9)
    ax.axis("off")
    H, W, _ = img.shape
    flat = img.reshape(-1, 3)
    for col, k in enumerate(ks, start=1):
        pal = paletas[k]
        est = pal[quantizar(flat, pal)].reshape(H, W, 3)
        ax = fig.add_subplot(gs[0, col]); ax.imshow(est); ax.axis("off")
        ax.set_title(f"k = {k}   RMSE {erros[k]:.1f}", fontsize=9)
        ax = fig.add_subplot(gs[1, col])
        ordem = np.argsort(pal.sum(1))            # da mais escura à mais clara
        ax.imshow(pal[ordem][None, :, :], aspect="auto"); ax.axis("off")
    fig.savefig("figs/p4_estilizadas.png", bbox_inches="tight")
    plt.close(fig)


def figura_erro_k(res):
    ks = sorted(res)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    for chave, cor, rot in (("ag", AZUL, "AG"), ("ag_mem", AQUA, "AG memético"),
                            ("kmeans", LARANJA, "k-means")):
        med = [np.mean(res[k][chave]["rmse"]) for k in ks]
        lo = [np.min(res[k][chave]["rmse"]) for k in ks]
        hi = [np.max(res[k][chave]["rmse"]) for k in ks]
        ax.plot(ks, med, "o-", color=cor, lw=1.6, ms=4, label=rot)
        ax.fill_between(ks, lo, hi, color=cor, alpha=0.15, lw=0)
    ax.plot(ks, [res[k]["uniforme"]["rmse"] for k in ks], "s--", color=TINTA2,
            lw=1.2, ms=4, label="paleta uniforme")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("Número de cores k", color=TINTA, fontsize=9)
    ax.set_ylabel("RMSE por canal (0-255)", color=TINTA, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    fig.savefig("figs/p4_erro_k.png", bbox_inches="tight")
    plt.close(fig)


def figura_convergencia(hist, k, ref_kmeans):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    g = np.arange(1, len(hist["ag"]["melhor"]) + 1)
    ax.plot(g, rmse255(np.array(hist["ag"]["media"])), color=AZUL, lw=1, alpha=0.5,
            label="AG: média da população")
    ax.plot(g, rmse255(np.array(hist["ag"]["melhor"])), color=AZUL, lw=1.6, label="AG: melhor")
    ax.plot(g, rmse255(np.array(hist["ag_mem"]["melhor"])), color=AQUA, lw=1.6,
            label="AG memético: melhor")
    ax.axhline(ref_kmeans, color=LARANJA, ls="--", lw=1.2, label="k-means (melhor de 10)")
    ax.set_xlabel("Geração", color=TINTA, fontsize=9)
    ax.set_ylabel("RMSE por canal (0-255)", color=TINTA, fontsize=9)
    ax.set_title(f"k = {k}", fontsize=9, color=TINTA2)
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    fig.savefig("figs/p4_convergencia.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    img = carregar_foto()
    todos = img.reshape(-1, 3)
    rng0 = np.random.default_rng(2026)
    amostra = todos[rng0.choice(len(todos), 4000, replace=False)]   # para a aptidão
    print(f"foto {img.shape[1]}x{img.shape[0]}, {len(todos)} pixels, amostra {len(amostra)}")

    KS, EXEC = (2, 4, 8, 16), 10
    res, paletas, erros_k, hist_k8 = {}, {}, {}, {}
    for k in KS:
        res[k] = {"ag": {"rmse": [], "t": []}, "ag_mem": {"rmse": [], "t": []},
                  "kmeans": {"rmse": [], "t": []}}
        melhor_ag = (np.inf, None)
        for e in range(EXEC):
            for chave, mem in (("ag", False), ("ag_mem", True)):
                rng = np.random.default_rng(100 * k + e)
                t0 = time.perf_counter()
                pal, hm, hme = AGPaleta(amostra, k, rng, memetico=mem).executar()
                res[k][chave]["t"].append(time.perf_counter() - t0)
                r = rmse255(erro(todos, pal))
                res[k][chave]["rmse"].append(r)
                if k == 8 and e == 0:
                    hist_k8[chave] = {"melhor": hm, "media": hme}
                if chave == "ag" and r < melhor_ag[0]:
                    melhor_ag = (r, pal)
            rng = np.random.default_rng(100 * k + e)
            t0 = time.perf_counter()
            pal = kmeans(amostra, k, rng)
            res[k]["kmeans"]["t"].append(time.perf_counter() - t0)
            res[k]["kmeans"]["rmse"].append(rmse255(erro(todos, pal)))
        res[k]["uniforme"] = {"rmse": rmse255(erro(todos, paleta_uniforme(k)))}
        paletas[k], erros_k[k] = melhor_ag[1], melhor_ag[0]
        print(f"k={k:2d}  AG {np.mean(res[k]['ag']['rmse']):6.2f} ± {np.std(res[k]['ag']['rmse']):.2f}"
              f"  AG-mem {np.mean(res[k]['ag_mem']['rmse']):6.2f} ± {np.std(res[k]['ag_mem']['rmse']):.2f}"
              f"  k-means {np.mean(res[k]['kmeans']['rmse']):6.2f} ± {np.std(res[k]['kmeans']['rmse']):.2f}"
              f"  uniforme {res[k]['uniforme']['rmse']:6.2f}"
              f"  | t AG {np.mean(res[k]['ag']['t']):.2f}s  k-means {np.mean(res[k]['kmeans']['t'])*1000:.0f}ms")

    figura_estilizadas(img, paletas, erros_k)
    figura_erro_k(res)
    figura_convergencia(hist_k8, 8, float(np.min(res[8]["kmeans"]["rmse"])))

    saida = {"imagem": "grace_hopper.jpg", "pixels": int(len(todos)), "amostra": int(len(amostra)),
             "execucoes": EXEC, "ks": list(KS), "resultados": {},
             "paletas_ag_255": {k: (paletas[k] * 255).round().astype(int).tolist() for k in KS}}
    for k in KS:
        saida["resultados"][k] = {
            ch: {"rmse_media": float(np.mean(v["rmse"])), "rmse_dp": float(np.std(v["rmse"])),
                 "rmse_min": float(np.min(v["rmse"])), "rmse_max": float(np.max(v["rmse"])),
                 "t_medio_s": float(np.mean(v["t"]))}
            for ch, v in res[k].items() if ch != "uniforme"}
        saida["resultados"][k]["uniforme"] = res[k]["uniforme"]
    with open("results/p4.json", "w") as f:
        json.dump(saida, f, indent=1)
    print("ok: figs/p4_*.png, results/p4.json")
