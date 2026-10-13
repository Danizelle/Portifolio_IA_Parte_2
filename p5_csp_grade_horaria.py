"""
Projeto 5 - Problema de satisfação de restrições (CSP): grade horária
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
Um curso precisa alocar as aulas da semana em horários e salas. Recursos:
  * 4 dias x 3 períodos = 12 horários;
  * 3 salas com capacidades 30, 45 e 60 alunos;
  * 6 professores e 4 turmas (semestres).
Cada disciplina tem 2 aulas semanais, um professor, uma turma e um número de
alunos matriculados.

Formulação como CSP
  Variáveis : cada aula (disciplina d, aula 1 ou 2).
  Domínios  : pares (horário, sala)  ->  12 x 3 = 36 valores.
  Restrições:
    R1 (unária)  capacidade da sala >= alunos matriculados;
    R2 (unária)  o professor não pode dar aula nos horários em que é
                 indisponível;
    R3 (binária) duas aulas não ocupam a mesma sala no mesmo horário;
    R4 (binária) aulas do mesmo professor ficam em horários diferentes;
    R5 (binária) aulas da mesma turma ficam em horários diferentes;
    R6 (binária) as duas aulas de uma disciplina ficam em dias diferentes.

Métodos comparados
  1. Backtracking puro (ordem fixa de variáveis e valores);
  2. Backtracking + MRV (menor número de valores restantes) com desempate
     pelo grau (variável mais restrita);
  3. Backtracking + MRV + LCV (valor menos restritivo);
  4. Backtracking + MRV + LCV + verificação adiante (forward checking);
  5. Backtracking + MRV + LCV + MAC (manutenção de consistência de arco
     com AC-3 após cada atribuição);
  6. Mínimos conflitos (busca local), como contraponto.
Antes da busca, as restrições unárias são aplicadas aos domínios
(consistência de nó) e, nos métodos 4 e 5, roda-se também o AC-3 inicial.

Execução:  python p5_csp_grade_horaria.py
Saídas:    figs/p5_*.png  e  results/p5.json
"""

import json
import os
import random
import statistics
import time
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIAS, PERIODOS = 4, 3
NOMES_DIAS = ["Seg", "Ter", "Qua", "Qui"]
NOMES_PER = ["08h-10h", "10h-12h", "14h-16h"]
SALAS = [30, 45, 60]                       # capacidades
H = DIAS * PERIODOS                        # 12 horários
N_PROF, N_TURMAS = 6, 4

# Bits que descrevem a relação entre duas aulas
MESMO_HORARIO_PROIBIDO = 1                 # mesmo professor ou mesma turma
DIA_DIFERENTE = 2                          # mesma disciplina


class OrcamentoEsgotado(Exception):
    """Interrompe a busca quando o limite de atribuições é atingido."""


# --------------------------------------------------------------------------
# Geração de instâncias
# --------------------------------------------------------------------------
def gerar_instancia(n_disc, seed):
    rng = random.Random(seed)
    # professores distribuídos de forma equilibrada (no máx. ceil(n/6) cada)
    profs = [d % N_PROF for d in range(n_disc)]
    rng.shuffle(profs)
    disciplinas = []
    for d in range(n_disc):
        disciplinas.append({
            "nome": f"D{d+1:02d}",
            "turma": d % N_TURMAS,
            "prof": profs[d],
            "alunos": rng.choice([20, 25, 30, 35, 40, 45, 50, 55]),
        })
    indisp = {p: set(rng.sample(range(H), 2)) for p in range(N_PROF)}
    return disciplinas, indisp


class CSPGrade:
    def __init__(self, disciplinas, indisp):
        self.disc = disciplinas
        # variáveis: (índice da disciplina, número da aula)
        self.vars = [(d, k) for d in range(len(disciplinas)) for k in (0, 1)]
        n = len(self.vars)
        # domínio completo: valor = (horário, sala)
        todos = [(h, s) for h in range(H) for s in range(len(SALAS))]
        # consistência de nó: aplica R1 e R2 diretamente nos domínios
        self.dom0 = []
        for d, _ in self.vars:
            info = disciplinas[d]
            self.dom0.append([(h, s) for h, s in todos
                              if SALAS[s] >= info["alunos"]
                              and h not in indisp[info["prof"]]])
        # matriz de relações entre pares de variáveis (R3 vale para todos)
        self.rel = [[0] * n for _ in range(n)]
        for i, (di, _) in enumerate(self.vars):
            for j, (dj, _) in enumerate(self.vars):
                if i == j:
                    continue
                a, b = disciplinas[di], disciplinas[dj]
                r = 0
                if a["prof"] == b["prof"] or a["turma"] == b["turma"]:
                    r |= MESMO_HORARIO_PROIBIDO
                if di == dj:
                    r |= DIA_DIFERENTE
                self.rel[i][j] = r
        # grafo de restrições: como R3 liga todos os pares, é completo
        self.vizinhos = [[j for j in range(n) if j != i] for i in range(n)]
        self.grau_forte = [sum(1 for j in range(n) if self.rel[i][j])
                           for i in range(n)]

    def compativel(self, i, a, j, b):
        """True se Xi = a e Xj = b satisfazem todas as restrições binárias."""
        if a == b:                                          # R3
            return False
        r = self.rel[i][j]
        if r & MESMO_HORARIO_PROIBIDO and a[0] == b[0]:     # R4, R5
            return False
        if r & DIA_DIFERENTE and a[0] // PERIODOS == b[0] // PERIODOS:  # R6
            return False
        return True

    def n_conflitos(self, i, a, atrib):
        return sum(1 for j, b in atrib.items()
                   if j != i and not self.compativel(i, a, j, b))


# --------------------------------------------------------------------------
# AC-3
# --------------------------------------------------------------------------
def revisar(csp, dom, i, j, remocoes):
    """Remove de Di os valores sem suporte em Dj. Retorna True se removeu."""
    removeu = False
    for a in dom[i][:]:
        if not any(csp.compativel(i, a, j, b) for b in dom[j]):
            dom[i].remove(a)
            remocoes.append((i, a))
            removeu = True
    return removeu


def ac3(csp, dom, fila, remocoes, atrib):
    fila = deque(fila)
    while fila:
        i, j = fila.popleft()
        if i in atrib:
            continue
        if revisar(csp, dom, i, j, remocoes):
            if not dom[i]:
                return False
            for k in csp.vizinhos[i]:
                if k != j and k not in atrib:
                    fila.append((k, i))
    return True


# --------------------------------------------------------------------------
# Backtracking genérico com heurísticas e inferência configuráveis
# --------------------------------------------------------------------------
def backtracking(csp, mrv=False, lcv=False, inferencia=None,
                 orcamento=2_000_000):
    n = len(csp.vars)
    dom = [list(d) for d in csp.dom0]
    stats = {"atribuicoes": 0, "retrocessos": 0}
    if inferencia in ("fc", "mac"):
        if not ac3(csp, dom, [(i, j) for i in range(n) for j in csp.vizinhos[i]],
                   [], {}):
            return None, stats
    atrib = {}

    def valores_legais(i):
        # sem inferência, o domínio não é podado: filtra pelo que já foi
        # atribuído (equivale a testar n_conflitos == 0, mas em O(n + |D|))
        if inferencia is None:
            ocupados, horarios_proib, dias_proib = set(), set(), set()
            for j, b in atrib.items():
                ocupados.add(b)
                r = csp.rel[i][j]
                if r & MESMO_HORARIO_PROIBIDO:
                    horarios_proib.add(b[0])
                if r & DIA_DIFERENTE:
                    dias_proib.add(b[0] // PERIODOS)
            return [a for a in dom[i] if a not in ocupados
                    and a[0] not in horarios_proib
                    and a[0] // PERIODOS not in dias_proib]
        return dom[i]

    def escolher_variavel():
        livres = [i for i in range(n) if i not in atrib]
        if not mrv:
            return livres[0]
        # MRV; desempate pela heurística de grau (mais restrições fortes)
        return min(livres, key=lambda i: (len(valores_legais(i)),
                                          -csp.grau_forte[i]))

    def ordenar_valores(i):
        vals = valores_legais(i) if inferencia is None else list(dom[i])
        if not lcv:
            return vals
        livres = [j for j in csp.vizinhos[i]
                  if j not in atrib and csp.rel[i][j]]
        # LCV: conta quantos valores o valor 'a' eliminaria dos vizinhos
        def custo(a):
            return sum(1 for j in livres for b in dom[j]
                       if not csp.compativel(i, a, j, b))
        return sorted(vals, key=custo)

    def bt():
        if len(atrib) == n:
            return True
        i = escolher_variavel()
        for a in ordenar_valores(i):
            stats["atribuicoes"] += 1
            if stats["atribuicoes"] > orcamento:
                raise OrcamentoEsgotado
            atrib[i] = a
            remocoes = []
            ok = True
            if inferencia in ("fc", "mac"):
                for b in dom[i][:]:            # fixa Di = {a}
                    if b != a:
                        dom[i].remove(b)
                        remocoes.append((i, b))
                if inferencia == "fc":
                    # verificação adiante: poda vizinhos não atribuídos
                    for j in csp.vizinhos[i]:
                        if j in atrib:
                            continue
                        for b in dom[j][:]:
                            if not csp.compativel(i, a, j, b):
                                dom[j].remove(b)
                                remocoes.append((j, b))
                        if not dom[j]:
                            ok = False
                            break
                else:
                    # MAC: AC-3 a partir dos arcos (vizinho, i)
                    ok = ac3(csp, dom, [(j, i) for j in csp.vizinhos[i]
                                        if j not in atrib], remocoes, atrib)
            if ok and bt():
                return True
            for j, b in remocoes:              # desfaz as podas
                dom[j].append(b)
            del atrib[i]
            stats["retrocessos"] += 1
        return False

    try:
        achou = bt()
    except OrcamentoEsgotado:
        stats["esgotou"] = True
        return None, stats
    return (dict(atrib) if achou else None), stats


def minimos_conflitos(csp, rng, passos_max=50_000):
    """Busca local: parte de uma atribuição completa aleatória e, a cada passo,
    escolhe uma variável em conflito e lhe dá o valor de menor conflito."""
    n = len(csp.vars)
    atrib = {i: rng.choice(csp.dom0[i]) for i in range(n)}
    for passo in range(passos_max):
        conflitantes = [i for i in range(n)
                        if csp.n_conflitos(i, atrib[i], atrib)]
        if not conflitantes:
            return atrib, passo
        i = rng.choice(conflitantes)
        custos = [(csp.n_conflitos(i, a, atrib), a) for a in csp.dom0[i]]
        menor = min(c for c, _ in custos)
        atrib[i] = rng.choice([a for c, a in custos if c == menor])
    return None, passos_max


def verificar(csp, atrib):
    """Confere independentemente todas as restrições de uma solução."""
    for i, a in atrib.items():
        if a not in csp.dom0[i]:
            return False
        for j, b in atrib.items():
            if i < j and not csp.compativel(i, a, j, b):
                return False
    return len(atrib) == len(csp.vars)


METODOS = [
    ("BT puro", dict()),
    ("BT + MRV", dict(mrv=True)),
    ("BT + MRV + LCV", dict(mrv=True, lcv=True)),
    ("BT + MRV + LCV + FC", dict(mrv=True, lcv=True, inferencia="fc")),
    ("BT + MRV + LCV + MAC", dict(mrv=True, lcv=True, inferencia="mac")),
]

TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    ORC = 50_000
    tamanhos = [12, 14, 16, 17, 18]           # disciplinas -> 24..36 aulas
    INST = 20
    res = {nome: {} for nome, _ in METODOS + [("Mínimos conflitos", None)]}
    descartadas = {}
    for nd in tamanhos:
        bruto = {nome: [] for nome in res}
        seed, aceitas, descartadas[2 * nd] = 1000 * nd, 0, 0
        while aceitas < INST:
            csp = CSPGrade(*gerar_instancia(nd, seed=seed))
            seed += 1
            # Certificação de satisfatibilidade: a instância só entra no
            # experimento se a busca local (mínimos conflitos, 3 reinícios de
            # 5 mil passos) achar uma solução. Instâncias sem solução conhecida
            # são descartadas, pois nenhum método consegue provar a
            # inviabilidade dentro do orçamento (ver relatório).
            rng = random.Random(seed)
            t_mc = time.perf_counter()
            passos_total, certificada = 0, None
            for _ in range(3):
                certificada, passos = minimos_conflitos(csp, rng, 5000)
                passos_total += passos
                if certificada is not None:
                    break
            if certificada is None:
                descartadas[2 * nd] += 1
                continue
            t_mc = time.perf_counter() - t_mc
            assert verificar(csp, certificada)
            aceitas += 1
            bruto["Mínimos conflitos"].append(
                {"atrib": passos_total, "bt": 0, "ok": True, "esgotou": False,
                 "t": t_mc})
            for nome, cfg in METODOS:
                t0 = time.perf_counter()
                sol, st = backtracking(csp, orcamento=ORC, **cfg)
                dt = time.perf_counter() - t0
                if sol is not None:
                    assert verificar(csp, sol)
                bruto[nome].append({"atrib": st["atribuicoes"],
                                    "bt": st["retrocessos"],
                                    "ok": sol is not None,
                                    "esgotou": st.get("esgotou", False),
                                    "t": dt})
        print(f"{2*nd} aulas: {descartadas[2*nd]} instâncias descartadas")
        for nome, linhas in bruto.items():
            res[nome][2 * nd] = {
                "mediana_atrib": statistics.median(l["atrib"] for l in linhas),
                "media_retrocessos": statistics.mean(l["bt"] for l in linhas),
                "resolvidas": sum(l["ok"] for l in linhas),
                "esgotou_orcamento": sum(l["esgotou"] for l in linhas),
                "tempo_med_ms": 1000 * statistics.mean(l["t"] for l in linhas),
            }
            r = res[nome][2 * nd]
            print(f"{2*nd} aulas | {nome:22s} med_atrib={r['mediana_atrib']:>9} "
                  f"resolv={r['resolvidas']}/{INST} esgot={r['esgotou_orcamento']} "
                  f"t={r['tempo_med_ms']:.1f} ms")

    # ---- Figura 1: (a) % resolvidas no orçamento  (b) mediana de atribuições
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=200)
    xs = [2 * nd for nd in tamanhos]
    nomes_bt = [nome for nome, _ in METODOS]
    for cor, nome in zip(CORES, nomes_bt):
        ys = [100 * res[nome][x]["resolvidas"] / INST for x in xs]
        ax1.plot(xs, ys, color=cor, linewidth=2, marker="o", markersize=4.5,
                 label=nome)
        ys = [res[nome][x]["mediana_atrib"] for x in xs]
        ax2.plot(xs, ys, color=cor, linewidth=2, marker="o", markersize=4.5)
    ax2.axhline(ORC, color=TINTA2, linestyle="--", linewidth=1)
    ax2.text(xs[0], ORC * 1.15, "orçamento: 50 mil", fontsize=7,
             color=TINTA2, va="bottom")
    ax2.set_yscale("log")
    ax1.set_ylabel("Instâncias resolvidas (%)", color=TINTA, fontsize=9)
    ax2.set_ylabel("Atribuições (mediana, log)", color=TINTA, fontsize=9)
    ax1.set_title("(a) sucesso dentro do orçamento", fontsize=8.5, color=TINTA)
    ax2.set_title("(b) esforço", fontsize=8.5, color=TINTA)
    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xlabel("Aulas a alocar (36 pares horário-sala)", color=TINTA,
                      fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=TINTA2, labelsize=8)
        ax.grid(axis="y", color=GRADE, linewidth=0.8)
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig("figs/p5_esforco.png")
    plt.close(fig)

    # ---- Figura 2: grade horária resolvida (maior instância) ----
    disc, indisp = gerar_instancia(16, seed=16000)
    csp = CSPGrade(disc, indisp)
    sol, st = backtracking(csp, mrv=True, lcv=True, inferencia="mac",
                           orcamento=10 ** 7)
    exemplo = {"resolvida": sol is not None, **st}
    if sol is not None:
        fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
        ax.set_xlim(0, DIAS); ax.set_ylim(PERIODOS, 0); ax.axis("off")
        cores_turma = ["#cde2fb", "#fbd9c9", "#c9eedf", "#fbe8bf"]
        for (d, k), (h, s) in ((csp.vars[i], v) for i, v in sol.items()):
            dia, per = divmod(h, PERIODOS)
            info = disc[d]
            x0, y0 = dia + 0.02 + s * 0.32, per + 0.06
            ax.add_patch(plt.Rectangle((x0, y0), 0.30, 0.86,
                         facecolor=cores_turma[info["turma"]],
                         edgecolor="#52514e", lw=0.6))
            ax.text(x0 + 0.15, y0 + 0.43,
                    f"{info['nome']}\nT{info['turma']+1}\nP{info['prof']+1}\n"
                    f"{info['alunos']} al.", ha="center", va="center",
                    fontsize=5.6, color=TINTA)
        for dia in range(DIAS):
            ax.text(dia + 0.5, -0.06, NOMES_DIAS[dia], ha="center",
                    fontsize=8.5, color=TINTA)
            for s, cap in enumerate(SALAS):
                ax.text(dia + 0.02 + s * 0.32 + 0.15, 3.07,
                        f"S{s+1}({cap})", ha="center", va="top",
                        fontsize=5.5, color=TINTA2)
        for per in range(PERIODOS):
            ax.text(-0.03, per + 0.5, NOMES_PER[per], ha="right",
                    va="center", fontsize=7.5, color=TINTA)
        fig.tight_layout()
        fig.savefig("figs/p5_grade_solucao.png")
        plt.close(fig)

    with open("results/p5.json", "w", encoding="utf-8") as f:
        json.dump({"orcamento": ORC, "instancias_por_tamanho": INST,
                   "resultados": res, "descartadas": descartadas,
                   "exemplo_32_aulas": exemplo,
                   "indisp_exemplo": {p: sorted(v) for p, v in indisp.items()}},
                  f, ensure_ascii=False, indent=1)
