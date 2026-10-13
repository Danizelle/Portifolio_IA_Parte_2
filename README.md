<div align="center">

# Métodos Clássicos de Inteligência Artificial
**FGA0221 — Inteligência Artificial**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557c?style=for-the-badge)](https://matplotlib.org/)
[![UnB](https://img.shields.io/badge/UnB-FCTE_Gama-003366?style=for-the-badge)](https://fga.unb.br/)

<br>

| Informação | Detalhe |
| :--- | :--- |
| **Instituição** | Universidade de Brasília (UnB) — Campus Gama |
| **Faculdade** | Faculdade de Ciências e Tecnologias em Engenharia (FCTE) |
| **Autora** | Danielle Rodrigues Silva |
| **Matrícula** | 211061574 |
| **Professor** | Fabiano Araujo Soares |


</div>

Este repositório contém os seis projetos práticos desenvolvidos para a parte 2 do repositório de Inteligência Artificial, abordando busca informada, busca não informada, busca local, algoritmos genéticos, problemas de satisfação de restrições (CSP) e bases de conhecimento.

---

## Projetos e Tempos de Execução

Os tempos foram medidos em ambiente de referência com Python 3.10+ (NumPy e Matplotlib):

| Projeto | Arquivo | Algoritmos | Tempo Estimado |
| :--- | :--- | :--- | :---: |
| 1. Busca Informada | [`p1_busca_informada_cao_guia.py`](p1_busca_informada_cao_guia.py) | A\*, UCS, A\* Ponderado ($w=2$), Gulosa | ~2 s |
| 2. Busca Não Informada | [`p2_busca_nao_informada_robo.py`](p2_busca_nao_informada_robo.py) | BFS, DFS, UCS, IDS | ~52 s |
| 3. Busca Complexa | [`p3_busca_complexa_melodia.py`](p3_busca_complexa_melodia.py) | Subida de encosta, Reinício aleatório, Têmpera simulada | ~8 s |
| 4. Algoritmo Genético | [`p4_algoritmo_genetico_paleta.py`](p4_algoritmo_genetico_paleta.py) | AG contínuo, AG memético, k-means | ~3,5 min |
| 5. CSP | [`p5_csp_grade_horaria.py`](p5_csp_grade_horaria.py) | Backtracking, MRV, LCV, FC, MAC (AC-3), Mínimos conflitos | ~4 min |
| 6. Base de Conhecimento | [`p6_base_conhecimento_tecnicas_artisticas.py`](p6_base_conhecimento_tecnicas_artisticas.py) | Encadeamento para frente (*PL-FC-Entails* trivalente) | ~2 s |

---

## Requisitos e Instalação

### Pré-requisitos
- Python 3.8 ou superior (testado em 3.10 e 3.13)
- pip

### Instalação

1. (Opcional) Crie e ative um ambiente virtual:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
   *(Dependências externas: `numpy` e `matplotlib`)*

---

## Instruções de Execução

Cada script cria automaticamente as pastas `figs/` e `results/` ao ser executado.

### Projeto 1 — Busca Informada: Cão-guia Virtual em Prédio
Roteamento em grade $40 \times 60$ ($0{,}5\text{ m}$ por célula) com penalização de curvas e preferência por piso tátil. Avalia 30 rotas e gera instruções textuais de navegação.

```bash
python p1_busca_informada_cao_guia.py
```
- **Saídas:** `figs/p1_rotas.png`, `figs/p1_esforco.png` e `results/p1.json`.

---

### Projeto 2 — Busca Não Informada: Robô de Entrega em Terrenos
Navegação em grade $30 \times 30$ com terrenos de custos variados (piso = 1, grama = 3, lama = 8) e análise de escalabilidade em mapas de $10\times10$ a $60\times60$.

```bash
python p2_busca_nao_informada_robo.py
```
- **Saídas:** `figs/p2_caminhos.png`, `figs/p2_escala.png` e `results/p2.json`.

---

### Projeto 3 — Busca em Ambientes Complexos: Composição Musical
Composição de uma melodia de 16 notas avaliada por 6 regras de harmonia e estrutura musical (máximo de 13 pontos), gerando arquivos de áudio sintetizados.

```bash
python p3_busca_complexa_melodia.py
```
- **Saídas:** `figs/p3_piano_roll.png`, `figs/p3_evolucao.png`, `results/p3.json`, `results/p3_melodia_inicial.wav` e `results/p3_melodia_final.wav`.

---

### Projeto 4 — Algoritmo Genético: Paleta de Cores e Quantização
Quantização da fotografia `grace_hopper.jpg` para $k \in \{2, 4, 8, 16\}$ cores, comparando AG real, AG memético (com passo de Lloyd) e k-means.

```bash
python p4_algoritmo_genetico_paleta.py
```
- **Saídas:** `figs/p4_estilizadas.png`, `figs/p4_erro_k.png`, `figs/p4_convergencia_k8.png` e `results/p4.json`.

---

### Projeto 5 — Satisfação de Restrições (CSP): Grade Horária
Alocação de aulas semanais em horários e salas sob restrições unárias (capacidade da sala, indisponibilidade) e binárias (conflito de sala, professor, turma e distribuição dos dias).

```bash
python p5_csp_grade_horaria.py
```
- **Saídas:** `figs/p5_esforco.png`, `figs/p5_grade_solucao.png` e `results/p5.json`.

---

### Projeto 6 — Base de Conhecimento: Assistente de Técnicas Artísticas
Sistema especialista baseado em 127 cláusulas de Horn e lógica trivalente (verdadeiro, falso, desconhecido), identificando técnicas recomendadas, materiais faltantes e gerando perguntas quando necessário.

- **Modo teste (avaliação de 30 cenários):**
  ```bash
  python p6_base_conhecimento_tecnicas_artisticas.py
  ```
- **Modo interativo (perguntas no terminal):**
  ```bash
  python p6_base_conhecimento_tecnicas_artisticas.py --interativo
  ```
- **Saídas:** `figs/p6_regras.png`, `figs/p6_avaliacao.png`, `results/p6.json` e `results/p6_exemplos.txt`.

---

## Execução em Lote

Para rodar todos os seis experimentos em sequência:

### Windows (PowerShell)
```powershell
python p1_busca_informada_cao_guia.py; python p2_busca_nao_informada_robo.py; python p3_busca_complexa_melodia.py; python p4_algoritmo_genetico_paleta.py; python p5_csp_grade_horaria.py; python p6_base_conhecimento_tecnicas_artisticas.py
```

### Linux / macOS (Bash)
```bash
python3 p1_busca_informada_cao_guia.py && \
python3 p2_busca_nao_informada_robo.py && \
python3 p3_busca_complexa_melodia.py && \
python3 p4_algoritmo_genetico_paleta.py && \
python3 p5_csp_grade_horaria.py && \
python3 p6_base_conhecimento_tecnicas_artisticas.py
```

---

## Estrutura de Arquivos

```text
Trabalho2/
├── p1_busca_informada_cao_guia.py
├── p2_busca_nao_informada_robo.py
├── p3_busca_complexa_melodia.py
├── p4_algoritmo_genetico_paleta.py
├── p5_csp_grade_horaria.py
├── p6_base_conhecimento_tecnicas_artisticas.py
├── requirements.txt
├── README.md
├── figs/       # Gráficos e figuras geradas (.png)
└── results/    # Relatórios (.json, .txt) e áudios sintetizados (.wav)
```

---

## Referências

- BUCHANAN, B. G.; SHORTLIFFE, E. H. *Rule-based expert systems*. Addison-Wesley, 1984.
- HART, P. E.; NILSSON, N. J.; RAPHAEL, B. A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 1968.
- HOLLAND, J. H. *Adaptation in natural and artificial systems*. University of Michigan Press, 1975.
- KIRKPATRICK, S.; GELATT, C. D.; VECCHI, M. P. Optimization by simulated annealing. *Science*, 1983.
- LLOYD, S. Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 1982.
- MACKWORTH, A. K. Consistency in networks of relations. *Artificial Intelligence*, 1977.
- MINTON, S. et al. Minimizing conflicts: a heuristic repair method for constraint-satisfaction and scheduling problems. *Artificial Intelligence*, 1992.
- POHL, I. Heuristic search viewed as path finding in a graph. *Artificial Intelligence*, 1970.
- RUSSELL, S.; NORVIG, P. *Inteligência Artificial: uma abordagem moderna*. 4. ed. GEN LTC, 2021.
