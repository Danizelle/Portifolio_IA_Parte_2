"""
Projeto 6 - Base de conhecimento: assistente de técnicas artísticas
FGA0221 - Inteligência Artificial - UnB, Campus Gama

Problema
--------
A pessoa informa o que tem e o que pretende criar:
  * materiais disponíveis  (lápis, carvão, aquarela, tinta acrílica, pincel...)
  * suporte                (papel próprio para aquarela, papel de desenho, tela)
  * efeito desejado        (transparência, cobertura opaca, linhas detalhadas,
                            esfumado)
  * preferências           (monocromático, colorido, combinação de técnicas)
e o sistema responde, com justificativa:
  * técnicas RECOMENDADAS;
  * ALTERNATIVAS que exigem material adicional (dizendo o que falta);
  * técnicas INCOMPATÍVEIS com as condições informadas (dizendo a restrição);
  * PERGUNTAS, quando falta informação para concluir.

Arquitetura (agente baseado em conhecimento)
  * Base de fatos:   o que o usuário informou, com TRÊS valores
                     (V verdadeiro, F falso, ? desconhecido);
  * Base de regras:  gerada a partir de uma tabela declarativa de técnicas
                     (TECNICAS, abaixo). Acrescentar uma técnica é acrescentar
                     uma entrada na tabela; o motor não muda;
  * Motor de inferência: encadeamento para frente sobre cláusulas de Horn
                     (algoritmo PL-FC-ENTAILS de Russell e Norvig), executado
                     duas vezes: uma tratando "?" como falso (o que está
                     PROVADO) e outra tratando "?" como verdadeiro (o que é
                     POSSÍVEL). Um átomo é V se provado, F se nem possível,
                     ? caso contrário. É assim que o sistema distingue
                     "incompatível" de "não sei" e decide o que perguntar.

Cadeia típica de inferência (para cada técnica t):
  tem(aquarela) + tem(pincel)            -> materiais_ok(t)
  materiais_ok(t) + suporte_ok(t)        -> viavel(t)
  efeito_ok(t, e) para todo efeito e     -> atende(t)
  viavel(t) + atende(t)                  -> adequada(t)
  adequada(t) + cor_ok(t) + comb_ok(t)   -> recomendada(t)

Alternativas por raciocínio hipotético: se recomendada(t) não é provada, o
sistema supõe que os materiais que faltam a t estão disponíveis e repete a
inferência; se t passa a ser recomendada (ou depende só de informação
faltante), ela é uma alternativa, e a diferença é a lista de compras.

Conhecimento: regras simplificadas de manuais de materiais artísticos
(MAYER, 1991; GOTTSEGEN, 2006; SMITH, 1987) - ver campo "fonte" e o
relatório. As regras são deliberadamente conservadoras.

Execução:
  python p6_base_conhecimento_tecnicas_artisticas.py              # experimento
  python p6_base_conhecimento_tecnicas_artisticas.py --interativo # perguntas no terminal
Saídas: figs/p6_*.png, results/p6.json, results/p6_exemplos.txt
"""

import argparse
import json
import os
import sys
import time
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

V, F, D = "V", "F", "?"          # verdadeiro, falso, desconhecido

# --------------------------------------------------------------------------
# Vocabulário
# --------------------------------------------------------------------------
MATERIAIS = {
    "lapis": "lápis grafite", "carvao": "carvão", "aquarela": "aquarela",
    "acrilica": "tinta acrílica", "pincel": "pincel", "nanquim": "nanquim",
    "bico_de_pena": "bico de pena", "medium_acrilico": "médium acrílico",
}
SUPORTES = {
    "papel_aquarela": "papel próprio para aquarela",
    "papel_desenho": "papel de desenho",
    "tela_preparada": "tela preparada",
}
EFEITOS = {
    "transparencia": "transparência", "opaca": "cobertura opaca",
    "linhas": "linhas detalhadas", "esfumado": "esfumado",
}
CORES = {"monocromatico": "monocromático", "colorido": "colorido"}

# --------------------------------------------------------------------------
# Base de conhecimento declarativa: uma entrada por técnica
#   materiais: todos necessários
#   suportes:  suporte -> (aceito?, justificativa)
#   produz:    efeitos que a técnica produz
#   cores:     modos de cor possíveis
#   mista:     combina duas técnicas
# --------------------------------------------------------------------------
TECNICAS = {
    "lavagem": {
        "curto": "lavagens",
        "nome": "Lavagens (aguadas) de aquarela",
        "materiais": ["aquarela", "pincel"],
        "suportes": {
            "papel_aquarela": (True, "papel encorpado e encolado, absorve a água sem deformar"),
            "papel_desenho": (False, "papel fino de desenho enruga e ondula com a água das aguadas"),
            "tela_preparada": (False, "a base acrílica da tela não absorve a aquarela, que escorre e não fixa"),
        },
        "produz": ["transparencia"], "cores": ["monocromatico", "colorido"], "mista": False,
        "fonte": "Mayer (1991); Smith (1987)",
    },
    "molhado": {
        "curto": "molhado sobre molhado",
        "nome": "Aquarela molhado sobre molhado",
        "materiais": ["aquarela", "pincel"],
        "suportes": {
            "papel_aquarela": (True, "papel pesado suporta ser encharcado antes da tinta"),
            "papel_desenho": (False, "papel fino de desenho se desfaz ou ondula quando encharcado"),
            "tela_preparada": (False, "a base acrílica da tela não absorve a aquarela"),
        },
        "produz": ["esfumado", "transparencia"], "cores": ["monocromatico", "colorido"], "mista": False,
        "fonte": "Mayer (1991); Smith (1987)",
    },
    "grafite": {
        "curto": "grafite",
        "nome": "Desenho de linha e hachura a grafite",
        "materiais": ["lapis"],
        "suportes": {
            "papel_desenho": (True, "superfície lisa a média segura o traço fino do grafite"),
            "papel_aquarela": (True, "aceita grafite, embora a textura reduza o detalhe"),
            "tela_preparada": (False, "na tela o grafite serve só de esboço: borra, brilha e não tem fixação"),
        },
        "produz": ["linhas"], "cores": ["monocromatico"], "mista": False,
        "fonte": "Smith (1987)",
    },
    "carvao": {
        "curto": "carvão",
        "nome": "Esfumado a carvão",
        "materiais": ["carvao"],
        "suportes": {
            "papel_desenho": (True, "papel com textura (dente) retém o pó do carvão"),
            "papel_aquarela": (True, "a textura do papel de aquarela retém bem o carvão"),
            "tela_preparada": (False, "na tela o carvão serve só de esboço sob a pintura; solta sem fixação"),
        },
        "produz": ["esfumado"], "cores": ["monocromatico"], "mista": False,
        "fonte": "Mayer (1991); Smith (1987)",
    },
    "nanquim": {
        "curto": "nanquim",
        "nome": "Desenho a nanquim com bico de pena",
        "materiais": ["nanquim", "bico_de_pena"],
        "suportes": {
            "papel_desenho": (True, "superfície lisa evita que a pena enrosque e a tinta se espalhe"),
            "papel_aquarela": (True, "aceita nanquim, com traço menos nítido na textura"),
            "tela_preparada": (False, "a trama da tela impede o traço fino da pena"),
        },
        "produz": ["linhas"], "cores": ["monocromatico"], "mista": False,
        "fonte": "Smith (1987)",
    },
    "acrilica_opaca": {
        "curto": "acrílica opaca",
        "nome": "Pintura acrílica em camadas opacas",
        "materiais": ["acrilica", "pincel"],
        "suportes": {
            "tela_preparada": (True, "suporte próprio para tinta acrílica"),
            "papel_aquarela": (True, "papel pesado suporta a tinta acrílica sem deformar"),
            "papel_desenho": (False, "papel fino ondula e a camada espessa de tinta deforma o suporte"),
        },
        "produz": ["opaca"], "cores": ["monocromatico", "colorido"], "mista": False,
        "fonte": "Gottsegen (2006)",
    },
    "veladura": {
        "curto": "veladura",
        "nome": "Veladuras com acrílica",
        "materiais": ["acrilica", "pincel", "medium_acrilico"],
        "suportes": {
            "tela_preparada": (True, "suporte próprio para tinta acrílica"),
            "papel_aquarela": (True, "papel pesado suporta as camadas de veladura"),
            "papel_desenho": (False, "papel fino ondula com as camadas úmidas de tinta"),
        },
        "produz": ["transparencia"], "cores": ["monocromatico", "colorido"], "mista": False,
        "fonte": "Gottsegen (2006): diluir a acrílica só com água enfraquece o filme; usa-se médium",
    },
    "mista": {
        "curto": "mista",
        "nome": "Técnica mista: aquarela sobre desenho a lápis",
        "materiais": ["aquarela", "pincel", "lapis"],
        "suportes": {
            "papel_aquarela": (True, "único suporte que aceita o lápis e as aguadas sobre ele"),
            "papel_desenho": (False, "papel fino de desenho enruga com as aguadas"),
            "tela_preparada": (False, "a base acrílica da tela não absorve a aquarela"),
        },
        "produz": ["transparencia", "linhas"], "cores": ["colorido"], "mista": True,
        "fonte": "Smith (1987)",
    },
}


# --------------------------------------------------------------------------
# Base de regras: compilada a partir da tabela (o motor não conhece técnicas)
# --------------------------------------------------------------------------
class Regra:
    def __init__(self, rid, corpo, cabeca, tecnica, tipo):
        self.id, self.corpo, self.cabeca = rid, tuple(corpo), cabeca
        self.tecnica, self.tipo = tecnica, tipo


def compilar_regras(tecnicas):
    regras = []

    def add(corpo, cabeca, t, tipo):
        regras.append(Regra(f"R{len(regras) + 1:03d}", corpo, cabeca, t, tipo))

    for t, d in tecnicas.items():
        add([f"tem({m})" for m in d["materiais"]], f"materiais_ok({t})", t, "materiais")
        aceitos = [s for s, (ok, _) in d["suportes"].items() if ok]
        for s in aceitos:
            add([f"suporte({s})"], f"suporte_ok({t})", t, "suporte")
        if {"papel_aquarela", "papel_desenho"} <= set(aceitos):
            # aceita qualquer papel: não precisa saber o tipo
            add(["suporte(papel)"], f"suporte_ok({t})", t, "suporte")
        add([f"materiais_ok({t})", f"suporte_ok({t})"], f"viavel({t})", t, "viabilidade")
        for e in EFEITOS:
            add([f"nao_deseja({e})"], f"efeito_ok({t},{e})", t, "efeito")
            if e in d["produz"]:
                add([f"deseja({e})"], f"efeito_ok({t},{e})", t, "efeito")
        add([f"efeito_ok({t},{e})" for e in EFEITOS], f"atende({t})", t, "atende")
        add([f"viavel({t})", f"atende({t})"], f"adequada({t})", t, "adequacao")
        add(["cor(indiferente)"], f"cor_ok({t})", t, "preferencia")
        for c in d["cores"]:
            add([f"cor({c})"], f"cor_ok({t})", t, "preferencia")
        add(["aceita(tecnica_unica)"], f"comb_ok({t})", t, "preferencia")
        if d["mista"]:
            add([], f"comb_ok({t})", t, "preferencia")
        add([f"adequada({t})", f"cor_ok({t})", f"comb_ok({t})"], f"recomendada({t})", t, "recomendacao")
    return regras


REGRAS = compilar_regras(TECNICAS)


# --------------------------------------------------------------------------
# Base de fatos: entrada do usuário -> valores V/F/?
# --------------------------------------------------------------------------
def montar_fatos(entrada):
    """entrada: dict com materiais (lista), suporte (None | 'papel' | lista),
    efeitos (None | lista), cor (None | 'monocromatico' | 'colorido'),
    combinacao (bool). Materiais: mundo fechado (o que não foi listado não
    está disponível). Suporte e efeito: desconhecidos se não informados."""
    fatos, nao_reconhecidos = {}, []
    mats = entrada.get("materiais", [])
    for m in MATERIAIS:
        fatos[f"tem({m})"] = V if m in mats else F
    nao_reconhecidos += [m for m in mats if m not in MATERIAIS]

    sup = entrada.get("suporte")
    if sup is None:
        for s in SUPORTES:
            fatos[f"suporte({s})"] = D
        fatos["suporte(papel)"] = D
    elif sup == "papel":                         # papel de tipo não informado
        fatos["suporte(papel_aquarela)"] = D
        fatos["suporte(papel_desenho)"] = D
        fatos["suporte(tela_preparada)"] = F
        fatos["suporte(papel)"] = V
    else:
        for s in SUPORTES:
            fatos[f"suporte({s})"] = V if s in sup else F
        fatos["suporte(papel)"] = V if {"papel_aquarela", "papel_desenho"} & set(sup) else F
        nao_reconhecidos += [s for s in sup if s not in SUPORTES]

    efs = entrada.get("efeitos")
    for e in EFEITOS:
        if efs is None:
            fatos[f"deseja({e})"] = fatos[f"nao_deseja({e})"] = D
        else:
            fatos[f"deseja({e})"] = V if e in efs else F
            fatos[f"nao_deseja({e})"] = F if e in efs else V
    if efs:
        nao_reconhecidos += [e for e in efs if e not in EFEITOS]

    cor = entrada.get("cor")
    fatos["cor(indiferente)"] = V if cor is None else F
    for c in CORES:
        fatos[f"cor({c})"] = V if cor == c else F
    fatos["aceita(tecnica_unica)"] = F if entrada.get("combinacao") else V
    return fatos, nao_reconhecidos


# --------------------------------------------------------------------------
# Motor de inferência
# --------------------------------------------------------------------------
def encadear_para_frente(verdadeiros, regras):
    """PL-FC-ENTAILS (Russell e Norvig): agenda de átomos, contador de
    premissas não satisfeitas por regra. Devolve os átomos derivados, a
    regra que provou cada um e a ordem em que as regras dispararam."""
    falta = {r.id: len(r.corpo) for r in regras}
    usos = {}
    for r in regras:
        for p in r.corpo:
            usos.setdefault(p, []).append(r)
    provados, prova, disparos = set(), {}, []
    agenda = deque(verdadeiros)
    for r in regras:                             # regras sem premissas (fatos da base)
        if not r.corpo:
            agenda.append(r.cabeca); prova.setdefault(r.cabeca, r); disparos.append(r.id)
    while agenda:
        a = agenda.popleft()
        if a in provados:
            continue
        provados.add(a)
        for r in usos.get(a, []):
            falta[r.id] -= 1
            if falta[r.id] == 0:
                disparos.append(r.id)
                if r.cabeca not in provados and r.cabeca not in prova:
                    prova[r.cabeca] = r
                agenda.append(r.cabeca)
    return provados, prova, disparos


class Inferencia:
    """Resultado da inferência em três valores sobre uma base de fatos."""

    def __init__(self, fatos, regras=REGRAS):
        self.fatos, self.regras = fatos, regras
        self.por_cabeca = {}
        for r in regras:
            self.por_cabeca.setdefault(r.cabeca, []).append(r)
        pess = [a for a, v in fatos.items() if v == V]
        otim = [a for a, v in fatos.items() if v in (V, D)]
        self.provados, self.prova, self.disparos = encadear_para_frente(pess, regras)
        self.possiveis, _, _ = encadear_para_frente(otim, regras)

    def valor(self, a):
        if a in self.fatos:
            return self.fatos[a]
        if a in self.provados:
            return V
        return D if a in self.possiveis else F

    def pendencias(self, a, vistos=None):
        """Fatos desconhecidos dos quais depende um átomo de valor '?'."""
        vistos = set() if vistos is None else vistos
        if a in vistos:
            return set()
        vistos.add(a)
        if a in self.fatos:
            return {a} if self.fatos[a] == D else set()
        out = set()
        for r in self.por_cabeca.get(a, []):
            if all(self.valor(p) != F for p in r.corpo):
                for p in r.corpo:
                    if self.valor(p) == D:
                        out |= self.pendencias(p, vistos)
        return out

    def motivos(self, a, vistos=None):
        """Fatos falsos que bloqueiam um átomo de valor F (todas as regras)."""
        vistos = set() if vistos is None else vistos
        if a in vistos:
            return set()
        vistos.add(a)
        if a in self.fatos:
            return {a} if self.fatos[a] == F else set()
        out = set()
        for r in self.por_cabeca.get(a, []):
            for p in r.corpo:
                if self.valor(p) == F:
                    out |= self.motivos(p, vistos)
        # efeito, cor e combinação: o motivo é o átomo intermediário, mais legível
        if a.startswith(("efeito_ok", "cor_ok", "comb_ok", "suporte_ok")):
            return {a}
        return out

    def arvore_prova(self, a, out=None, vistos=None):
        """Regras usadas para provar a, em pós-ordem (premissas antes)."""
        out = [] if out is None else out
        vistos = set() if vistos is None else vistos
        if a in vistos or a in self.fatos:
            return out
        vistos.add(a)
        r = self.prova.get(a)
        if r is None:
            return out
        for p in r.corpo:
            self.arvore_prova(p, out, vistos)
        out.append(r)
        return out


# --------------------------------------------------------------------------
# Classificação das técnicas e explicações
# --------------------------------------------------------------------------
def classificar(entrada):
    fatos, nao_rec = montar_fatos(entrada)
    inf = Inferencia(fatos)
    res = {"recomendadas": {}, "alternativas": {}, "incompativeis": {}, "indeterminadas": {},
           "perguntas": set(), "nao_reconhecidos": nao_rec,
           "regras_disparadas": len(inf.disparos), "inferencia": inf}
    for t, d in TECNICAS.items():
        alvo = f"recomendada({t})"
        if inf.valor(alvo) == V:
            res["recomendadas"][t] = inf.arvore_prova(alvo)
            continue
        faltam = [m for m in d["materiais"] if fatos[f"tem({m})"] == F]
        hip = dict(fatos)
        for m in faltam:                          # raciocínio hipotético
            hip[f"tem({m})"] = V
        inf_h = Inferencia(hip) if faltam else inf
        v_h = inf_h.valor(alvo)
        if v_h == F:
            res["incompativeis"][t] = {"motivos": sorted(inf_h.motivos(alvo)), "faltam": faltam}
        elif faltam:
            pend = sorted(inf_h.pendencias(alvo)) if v_h == D else []
            res["alternativas"][t] = {"faltam": faltam, "depende_de": pend}
        else:                                     # materiais presentes, falta informação
            pend = sorted(inf.pendencias(alvo))
            res["indeterminadas"][t] = pend
            res["perguntas"] |= {topico(p) for p in pend}
    return res


def topico(atomo):
    if atomo.startswith("suporte"):
        return "suporte"
    if atomo.startswith(("deseja", "nao_deseja")):
        return "efeito"
    return atomo


def arg(atomo):
    return atomo[atomo.index("(") + 1:-1]


def nome_atomo(a):
    """Nome legível de um átomo."""
    if a.startswith("tem("):
        return f"{MATERIAIS[arg(a)]} disponível"
    if a.startswith("suporte("):
        s = arg(a)
        return "suporte é papel" if s == "papel" else f"suporte: {SUPORTES[s]}"
    if a.startswith("deseja("):
        return f"deseja {EFEITOS[arg(a)]}"
    if a.startswith("nao_deseja("):
        return f"não pediu {EFEITOS[arg(a)]}"
    if a == "cor(indiferente)":
        return "sem preferência de cor"
    if a.startswith("cor("):
        return f"prefere {CORES[arg(a)]}"
    if a == "aceita(tecnica_unica)":
        return "aceita técnica única"
    t = arg(a).split(",")[0]
    n = TECNICAS[t]["curto"] if t in TECNICAS else t
    if a.startswith("efeito_ok("):
        return f"efeito {EFEITOS[arg(a).split(',')[1]]} atendido"
    return {"materiais_ok": f"materiais de «{n}» disponíveis",
            "suporte_ok": f"suporte adequado a «{n}»",
            "viavel": f"«{n}» viável", "atende": f"«{n}» atende aos efeitos",
            "adequada": f"«{n}» adequada", "cor_ok": "preferência de cor atendida",
            "comb_ok": "preferência de combinação atendida",
            "recomendada": f"«{n}» RECOMENDADA"}[a[:a.index("(")]]


def texto_regra(r):
    if r.tipo == "atende":     # 4 premissas efeito_ok: resume
        return f"{r.id}: todos os efeitos pedidos atendidos → {nome_atomo(r.cabeca)}"
    corpo = " + ".join(nome_atomo(p) for p in r.corpo) or "(técnica mista)"
    return f"{r.id}: {corpo} → {nome_atomo(r.cabeca)}"


def explicar_motivo(atomo, entrada):
    t = arg(atomo).split(",")[0]
    d = TECNICAS[t]
    if atomo.startswith("suporte_ok"):
        sup = entrada.get("suporte") or []
        frases = [f"{SUPORTES[s]}: {d['suportes'][s][1]}" for s in sup if s in d["suportes"]]
        return "suporte inadequado (" + "; ".join(frases) + ")"
    if atomo.startswith("efeito_ok"):
        e = arg(atomo).split(",")[1]
        return f"não produz {EFEITOS[e]} (produz: {', '.join(EFEITOS[x] for x in d['produz'])})"
    if atomo.startswith("cor_ok"):
        return f"é {' ou '.join(CORES[c] for c in d['cores'])}, e você prefere {CORES[entrada['cor']]}"
    if atomo.startswith("comb_ok"):
        return "você pediu combinação de técnicas e esta usa uma só"
    return nome_atomo(atomo)


PERGUNTAS = {
    "suporte": "Em que suporte você vai trabalhar: papel próprio para aquarela, papel de desenho ou tela preparada?",
    "suporte_papel": "Que tipo de papel? É papel próprio para aquarela (encorpado, cerca de 300 g/m²) ou papel de desenho comum?",
    "efeito": "Que efeito você deseja: transparência, cobertura opaca, linhas detalhadas ou esfumado?",
}


def lista_pt(itens):
    itens = list(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def relatorio_texto(entrada, res, cadeia=True):
    L = []
    mats = [MATERIAIS.get(m, m) for m in entrada.get("materiais", [])]
    sup = entrada.get("suporte")
    sup_txt = ("papel (tipo não informado)" if sup == "papel" else
               "suporte não informado" if sup is None else lista_pt(SUPORTES.get(s, s) for s in sup))
    efs = entrada.get("efeitos")
    ef_txt = "efeito não informado" if efs is None else "deseja " + lista_pt(EFEITOS.get(e, e) for e in efs)
    extra = []
    if entrada.get("cor"):
        extra.append(f"prefere {CORES[entrada['cor']]}")
    if entrada.get("combinacao"):
        extra.append("quer combinar técnicas")
    L.append(f"Você informou: {lista_pt(mats)}; {sup_txt}; {ef_txt}" + ("; " + "; ".join(extra) if extra else "") + ".")
    if res["nao_reconhecidos"]:
        L.append(f"Não está na base de conhecimento (ignorado): {', '.join(res['nao_reconhecidos'])}.")
    if res["recomendadas"]:
        L.append("SUGESTÃO")
        for t, prova in res["recomendadas"].items():
            d = TECNICAS[t]
            L.append(f"  • {d['nome']}")
            sups = [SUPORTES[s] for s in (sup if isinstance(sup, list) else []) if d["suportes"].get(s, (False,))[0]]
            efs_ok = [EFEITOS[e] for e in (efs or []) if e in d["produz"]]
            L.append(f"    Justificativa: os materiais necessários estão disponíveis ({lista_pt(MATERIAIS[m] for m in d['materiais'])}), "
                     f"o suporte foi considerado compatível ({lista_pt(sups) if sups else 'papel'}) "
                     f"e a técnica atende ao efeito desejado ({lista_pt(efs_ok) if efs_ok else '—'}).")
            if cadeia:
                L.append("    Cadeia de inferência:")
                L += [f"      {texto_regra(r)}" for r in prova if r.tipo != "efeito" or r.corpo and r.corpo[0].startswith("deseja")]
    if res["alternativas"]:
        L.append("ALTERNATIVAS (exigem material adicional)")
        for t, a in res["alternativas"].items():
            s = f"  • {TECNICAS[t]['nome']} — falta: {lista_pt(MATERIAIS[m] for m in a['faltam'])}"
            if a["depende_de"]:
                s += f"; depende também de: {lista_pt(sorted({topico(p) for p in a['depende_de']}))}"
            L.append(s)
    if res["incompativeis"]:
        L.append("INCOMPATÍVEIS com as condições informadas")
        for t, a in res["incompativeis"].items():
            mot = "; ".join(explicar_motivo(m, entrada) for m in a["motivos"])
            L.append(f"  • {TECNICAS[t]['nome']} — {mot}")
    if res["indeterminadas"]:
        L.append("SEM CONCLUSÃO: falta informação para " + lista_pt(TECNICAS[t]["nome"].lower() for t in res["indeterminadas"]))
        L.append("PERGUNTAS")
        for p in sorted(res["perguntas"]):
            chave = "suporte_papel" if p == "suporte" and sup == "papel" else p
            L.append(f"  ? {PERGUNTAS[chave]}")
    L.append(f"[{res['regras_disparadas']} regras disparadas de {len(REGRAS)}]")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Cenários de avaliação com resultado esperado definido previamente
#   rec: técnicas recomendadas; alt: alternativas {técnica: materiais que faltam};
#   perg: tópicos de pergunta. Grupo "fora da base": o gabarito segue o
#   conhecimento de arte, inclusive técnicas que a base não contém.
# --------------------------------------------------------------------------
def C(grupo, mats, sup, efs, rec=(), alt=None, perg=(), cor=None, comb=False, nota=""):
    return {"grupo": grupo, "entrada": {"materiais": list(mats), "suporte": sup, "efeitos": efs,
                                        "cor": cor, "combinacao": comb},
            "esperado": {"rec": set(rec), "alt": {k: set(v) for k, v in (alt or {}).items()},
                         "perg": set(perg)}, "nota": nota}


PA, PD, TL = ["papel_aquarela"], ["papel_desenho"], ["tela_preparada"]
CENARIOS = [
    # ---- completos: tudo informado, materiais suficientes ----
    C("completo", ["aquarela", "pincel"], PA, ["transparencia"], rec={"lavagem", "molhado"},
      alt={"mista": ["lapis"], "veladura": ["acrilica", "medium_acrilico"]}, nota="exemplo do enunciado"),
    C("completo", ["lapis"], PD, ["linhas"], rec={"grafite"}, alt={"nanquim": ["nanquim", "bico_de_pena"]}),
    C("completo", ["carvao"], PD, ["esfumado"], rec={"carvao"}, cor="monocromatico"),
    C("completo", ["acrilica", "pincel"], TL, ["opaca"], rec={"acrilica_opaca"}),
    C("completo", ["acrilica", "pincel", "medium_acrilico"], TL, ["transparencia"], rec={"veladura"}),
    C("completo", ["aquarela", "pincel", "lapis"], PA, ["transparencia", "linhas"], rec={"mista"}, comb=True),
    C("completo", ["nanquim", "bico_de_pena", "lapis"], PD, ["linhas"], rec={"nanquim", "grafite"}, cor="monocromatico"),
    C("completo", ["aquarela", "pincel"], PA, ["esfumado"], rec={"molhado"}, alt={"carvao": ["carvao"]}),
    C("completo", ["acrilica", "pincel"], PA, ["opaca"], rec={"acrilica_opaca"}, cor="colorido"),
    C("completo", ["aquarela", "pincel"], PA, ["transparencia"], rec={"lavagem", "molhado"},
      alt={"veladura": ["acrilica", "medium_acrilico"]}, cor="monocromatico"),
    # ---- materiais faltantes ----
    C("materiais faltantes", ["pincel"], PA, ["transparencia"],
      alt={"lavagem": ["aquarela"], "molhado": ["aquarela"], "veladura": ["acrilica", "medium_acrilico"],
           "mista": ["aquarela", "lapis"]}),
    C("materiais faltantes", ["acrilica", "pincel"], TL, ["transparencia"], alt={"veladura": ["medium_acrilico"]}),
    C("materiais faltantes", ["lapis"], PD, ["esfumado"], alt={"carvao": ["carvao"]}),
    C("materiais faltantes", ["aquarela"], PA, ["transparencia"],
      alt={"lavagem": ["pincel"], "molhado": ["pincel"], "veladura": ["acrilica", "pincel", "medium_acrilico"],
           "mista": ["pincel", "lapis"]}),
    C("materiais faltantes", ["nanquim"], PD, ["linhas"], alt={"nanquim": ["bico_de_pena"], "grafite": ["lapis"]}),
    C("materiais faltantes", ["carvao", "lapis"], TL, ["opaca"], alt={"acrilica_opaca": ["acrilica", "pincel"]}),
    # ---- informação insuficiente ----
    C("informação insuficiente", ["aquarela", "pincel"], "papel", ["transparencia"],
      alt={"mista": ["lapis"], "veladura": ["acrilica", "medium_acrilico"]}, perg={"suporte"}),
    C("informação insuficiente", ["aquarela", "pincel"], PA, None, perg={"efeito"},
      alt={"acrilica_opaca": ["acrilica"], "veladura": ["acrilica", "medium_acrilico"], "mista": ["lapis"],
           "grafite": ["lapis"], "carvao": ["carvao"], "nanquim": ["nanquim", "bico_de_pena"]}),
    C("informação insuficiente", ["lapis"], None, ["linhas"], perg={"suporte"},
      alt={"nanquim": ["nanquim", "bico_de_pena"], "mista": ["aquarela", "pincel"]}),
    C("informação insuficiente", ["carvao"], "papel", ["esfumado"], rec={"carvao"},
      alt={"molhado": ["aquarela", "pincel"]}, nota="aceita qualquer papel: não pergunta"),
    C("informação insuficiente", ["acrilica", "pincel"], TL, None, perg={"efeito"},
      alt={"veladura": ["medium_acrilico"]}),
    C("informação insuficiente", ["aquarela", "pincel", "lapis"], None, None, perg={"suporte", "efeito"},
      alt={"carvao": ["carvao"], "nanquim": ["nanquim", "bico_de_pena"], "acrilica_opaca": ["acrilica"],
           "veladura": ["acrilica", "medium_acrilico"]}),
    # ---- condições incompatíveis (suporte ou preferência) ----
    C("condições incompatíveis", ["aquarela", "pincel"], PD, ["transparencia"]),
    C("condições incompatíveis", ["lapis", "carvao"], PD, ["transparencia"]),
    C("condições incompatíveis", ["aquarela", "pincel"], TL, ["transparencia"],
      alt={"veladura": ["acrilica", "medium_acrilico"]}),
    C("condições incompatíveis", ["lapis"], PD, ["linhas"], cor="colorido"),
    C("condições incompatíveis", ["carvao"], PD, ["esfumado"], comb=True),
    # ---- fora da base: o gabarito inclui técnicas que a base não tem ----
    C("fora da base", ["guache", "pincel"], PA, ["opaca"], rec={"guache (fora da base)"},
      alt={"acrilica_opaca": ["acrilica"]}),
    C("fora da base", ["pastel_seco"], PD, ["esfumado"], rec={"pastel seco esfumado (fora da base)"},
      alt={"carvao": ["carvao"]}),
    C("fora da base", ["lapis_aquarelavel", "pincel"], PA, ["transparencia"],
      rec={"lápis aquarelável (fora da base)"},
      alt={"lavagem": ["aquarela"], "molhado": ["aquarela"], "veladura": ["acrilica", "medium_acrilico"],
           "mista": ["aquarela", "lapis"]}),
]


def avaliar(cen):
    res = classificar(cen["entrada"])
    obtido = {"rec": set(res["recomendadas"]),
              "alt": {t: set(a["faltam"]) for t, a in res["alternativas"].items()},
              "perg": set(res["perguntas"])}
    esp = cen["esperado"]
    ok = {k: obtido[k] == esp[k] for k in ("rec", "alt", "perg")}
    return res, obtido, ok


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
AZUL, LARANJA, AQUA, AMARELO, VERMELHO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
TINTA, TINTA2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
COR_VALOR = {V: AQUA, F: VERMELHO, D: AMARELO}


def estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=TINTA2, labelsize=8.5)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def desenhar_cadeia(ax, inf, t, titulo):
    """Grafo de inferência de uma técnica, com os nós coloridos pelo valor."""
    W, H = 0.17, 0.13
    nos = {   # átomo: (x, y, rótulo)
        "tem(aquarela)": (0.09, 0.90, "aquarela\ndisponível"),
        "tem(pincel)": (0.09, 0.72, "pincel\ndisponível"),
        "suporte(papel_aquarela)": (0.09, 0.50, "papel de\naquarela"),
        "deseja(transparencia)": (0.09, 0.30, "quer trans-\nparência"),
        "cor(indiferente)": (0.09, 0.10, "cor\nindiferente"),
        f"materiais_ok({t})": (0.32, 0.81, "materiais\ndisponíveis"),
        f"suporte_ok({t})": (0.32, 0.50, "suporte\nadequado"),
        f"atende({t})": (0.32, 0.30, "atende aos\nefeitos"),
        f"cor_ok({t})": (0.32, 0.10, "preferências\natendidas"),
        f"viavel({t})": (0.55, 0.66, "técnica\nviável"),
        f"adequada({t})": (0.74, 0.48, "técnica\nadequada"),
        f"recomendada({t})": (0.92, 0.29, "recomen-\ndada"),
    }
    arestas = [("tem(aquarela)", f"materiais_ok({t})"), ("tem(pincel)", f"materiais_ok({t})"),
               ("suporte(papel_aquarela)", f"suporte_ok({t})"), ("deseja(transparencia)", f"atende({t})"),
               ("cor(indiferente)", f"cor_ok({t})"),
               (f"materiais_ok({t})", f"viavel({t})"), (f"suporte_ok({t})", f"viavel({t})"),
               (f"viavel({t})", f"adequada({t})"), (f"atende({t})", f"adequada({t})"),
               (f"adequada({t})", f"recomendada({t})"), (f"cor_ok({t})", f"recomendada({t})")]
    for a, b in arestas:
        (x1, y1, _), (x2, y2, _) = nos[a], nos[b]
        ax.annotate("", xy=(x2 - W / 2, y2), xytext=(x1 + W / 2, y1),
                    arrowprops=dict(arrowstyle="-|>", color=TINTA2, lw=0.8, shrinkA=0, shrinkB=1,
                                    mutation_scale=7))
    for a, (x, y, rot) in nos.items():
        v = inf.valor(a)
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.004",
                                    fc=COR_VALOR[v], ec="none"))
        ax.text(x, y, rot, ha="center", va="center", fontsize=5.6, color="white" if v != D else TINTA,
                linespacing=1.1)
    ax.set_xlim(0, 1.0); ax.set_ylim(0.02, 0.98); ax.axis("off")
    ax.set_title(titulo, fontsize=8.5, color=TINTA2)


def figura_regras(entradas, titulos):
    fig, axes = plt.subplots(1, len(entradas), figsize=(10, 2.9), dpi=200)
    for ax, ent, tit in zip(axes, entradas, titulos):
        desenhar_cadeia(ax, classificar(ent)["inferencia"], "lavagem", tit)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=COR_VALOR[v]) for v in (V, D, F)]
    fig.legend(hs, ["provado (V)", "desconhecido (?): gera pergunta", "falso (F): gera restrição"],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.88, bottom=0.08)
    fig.savefig("figs/p6_regras.png", bbox_inches="tight")
    plt.close(fig)


def figura_avaliacao(linhas):
    grupos = list(dict.fromkeys(l["grupo"] for l in linhas))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=200, gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    acc = [100 * np.mean([l["correto"] for l in linhas if l["grupo"] == g]) for g in grupos]
    n = [sum(1 for l in linhas if l["grupo"] == g) for g in grupos]
    cores = [AZUL] * (len(grupos) - 1) + [LARANJA]
    ax.barh(range(len(grupos))[::-1], acc, color=cores, height=0.6)
    for i, (a, k) in enumerate(zip(acc, n)):
        ax.text(a + 2, len(grupos) - 1 - i, f"{a:.0f}% ({round(a * k / 100)}/{k})", va="center", fontsize=8, color=TINTA)
    ax.set_yticks(range(len(grupos))[::-1]); ax.set_yticklabels(grupos, fontsize=8)
    ax.set_xlim(0, 128); ax.set_xticks(range(0, 101, 20)); ax.set_xlabel("Cenários corretos (%)", color=TINTA, fontsize=9)
    ax.set_title("(a) acerto por grupo de cenários", fontsize=9, color=TINTA2)
    estilo(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRADE, lw=0.8)
    ax = axes[1]
    for i, g in enumerate(grupos):
        ys = [l["regras_disparadas"] for l in linhas if l["grupo"] == g]
        ax.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys)), ys, s=16,
                   color=cores[i], alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.mean(ys)] * 2, color=TINTA, lw=1.4)
    ax.set_xticks(range(len(grupos))); ax.set_xticklabels([g.replace(" ", "\n", 1) for g in grupos], fontsize=7.5)
    ax.set_ylabel(f"Regras disparadas (de {len(REGRAS)})", color=TINTA, fontsize=9)
    ax.set_title("(b) regras disparadas por cenário (traço = média)", fontsize=9, color=TINTA2)
    estilo(ax)
    fig.tight_layout()
    fig.savefig("figs/p6_avaliacao.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Modo interativo (perguntas no terminal)
# --------------------------------------------------------------------------
def escolher(titulo, opcoes, multipla=True, permite_nao_sei=False):
    print(f"\n{titulo}")
    chaves = list(opcoes)
    for i, k in enumerate(chaves, 1):
        print(f"  {i}. {opcoes[k]}")
    if permite_nao_sei:
        print("  0. não sei / não quero informar")
    txt = input("  números separados por vírgula: " if multipla else "  número: ").strip()
    if permite_nao_sei and txt in ("", "0"):
        return None
    idx = [int(x) for x in txt.replace(" ", "").split(",") if x.isdigit() and 0 < int(x) <= len(chaves)]
    return [chaves[i - 1] for i in idx]


def interativo():
    print("Assistente de técnicas artísticas — base de conhecimento com encadeamento para frente")
    entrada = {"materiais": escolher("Materiais disponíveis:", MATERIAIS)}
    sup = escolher("Suporte:", {**SUPORTES, "papel": "papel (não sei o tipo)"}, permite_nao_sei=True)
    entrada["suporte"] = "papel" if sup == ["papel"] else sup
    entrada["efeitos"] = escolher("Efeito desejado:", EFEITOS, permite_nao_sei=True)
    cor = escolher("Preferência de cor:", CORES, multipla=False, permite_nao_sei=True)
    entrada["cor"] = cor[0] if cor else None
    entrada["combinacao"] = input("\nQuer combinar técnicas? (s/N) ").strip().lower().startswith("s")
    while True:
        res = classificar(entrada)
        print("\n" + relatorio_texto(entrada, res))
        if not res["perguntas"]:
            break
        print("\nPara concluir, responda:")
        if "suporte" in res["perguntas"]:
            op = {k: v for k, v in SUPORTES.items() if not (entrada["suporte"] == "papel" and k == "tela_preparada")}
            entrada["suporte"] = escolher(PERGUNTAS["suporte_papel" if entrada["suporte"] == "papel" else "suporte"], op)
        if "efeito" in res["perguntas"]:
            entrada["efeitos"] = escolher(PERGUNTAS["efeito"], EFEITOS)


# --------------------------------------------------------------------------
# Experimento
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interativo", action="store_true")
    if ap.parse_args().interativo:
        interativo()
        raise SystemExit
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    tipos = {}
    for r in REGRAS:
        tipos[r.tipo] = tipos.get(r.tipo, 0) + 1
    print(f"base: {len(TECNICAS)} técnicas compiladas em {len(REGRAS)} regras {tipos}")

    linhas, textos = [], []
    t0 = time.perf_counter()
    for i, cen in enumerate(CENARIOS, 1):
        res, obt, ok = avaliar(cen)
        correto = all(ok.values())
        linhas.append({"id": i, "grupo": cen["grupo"], "correto": correto, "ok": ok,
                       "regras_disparadas": res["regras_disparadas"],
                       "obtido": {"rec": sorted(obt["rec"]), "alt": {k: sorted(v) for k, v in obt["alt"].items()},
                                  "perg": sorted(obt["perg"])},
                       "esperado": {"rec": sorted(cen["esperado"]["rec"]),
                                    "alt": {k: sorted(v) for k, v in cen["esperado"]["alt"].items()},
                                    "perg": sorted(cen["esperado"]["perg"])}})
        marca = "ok " if correto else "ERRO"
        print(f"[{marca}] C{i:02d} {cen['grupo']:24s} regras {res['regras_disparadas']:3d}  "
              f"rec={sorted(obt['rec'])} alt={sorted(obt['alt'])} perg={sorted(obt['perg'])}"
              + ("" if correto else f"  | esperado rec={sorted(cen['esperado']['rec'])} "
                                    f"alt={sorted(cen['esperado']['alt'])} perg={sorted(cen['esperado']['perg'])}"))
        textos.append(f"=== Cenário C{i:02d} ({cen['grupo']}{'; ' + cen['nota'] if cen['nota'] else ''}) ===\n"
                      + relatorio_texto(cen["entrada"], res, cadeia=(i in (1,))) + "\n")
    dt = (time.perf_counter() - t0) / len(CENARIOS)
    grupos = list(dict.fromkeys(l["grupo"] for l in linhas))
    resumo = {g: {"cenarios": sum(1 for l in linhas if l["grupo"] == g),
                  "corretos": sum(1 for l in linhas if l["grupo"] == g and l["correto"]),
                  "regras_media": float(np.mean([l["regras_disparadas"] for l in linhas if l["grupo"] == g]))}
              for g in grupos}
    dentro = [l for l in linhas if l["grupo"] != "fora da base"]
    print(f"\nacerto: {sum(l['correto'] for l in dentro)}/{len(dentro)} dentro da cobertura da base; "
          f"{sum(l['correto'] for l in linhas)}/{len(linhas)} no total; {dt*1000:.1f} ms por cenário")
    for g, r in resumo.items():
        print(f"  {g:24s} {r['corretos']}/{r['cenarios']}  regras disparadas (média) {r['regras_media']:.1f}")

    with open("results/p6_exemplos.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(textos))
    print("\n" + textos[0] + "\n" + textos[16] + "\n" + textos[19] + "\n" + textos[24])

    figura_regras([CENARIOS[0]["entrada"], CENARIOS[16]["entrada"], CENARIOS[24]["entrada"]],
                  ["(a) papel de aquarela (C01)", "(b) só \"papel\", sem o tipo (C17)",
                   "(c) tela preparada (C25)"])
    figura_avaliacao(linhas)
    with open("results/p6.json", "w", encoding="utf-8") as f:
        json.dump({"tecnicas": len(TECNICAS), "regras": len(REGRAS), "regras_por_tipo": tipos,
                   "cenarios": len(CENARIOS), "ms_por_cenario": dt * 1000, "resumo": resumo,
                   "acerto_dentro_base": [sum(l["correto"] for l in dentro), len(dentro)],
                   "acerto_total": [sum(l["correto"] for l in linhas), len(linhas)], "linhas": linhas},
                  f, indent=1, ensure_ascii=False)
    print("ok: figs/p6_*.png, results/p6.json, results/p6_exemplos.txt")
