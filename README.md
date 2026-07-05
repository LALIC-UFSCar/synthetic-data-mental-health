# Geração e Avaliação de Dados Sintéticos para Aplicações em Saúde Mental

Pipeline de paráfrase do corpus AMIVE via back-translation, permitindo sua disponibilização pública sem expor os textos originais.

O corpus AMIVE (*Análise de Manifestações de Inteligência e Vulnerabilidades Emocionais*) contém postagens de universitários em páginas anônimas de redes sociais, anotadas com sintomas de saúde mental. Por conter dados sensíveis e potencialmente identificáveis, o corpus original não pode ser distribuído publicamente. Este repositório implementa um pipeline de geração de dados sintéticos que preserva as características linguísticas e semânticas do corpus, viabilizando a distribuição de versões parafraseadas.

---

## 🎯 Objetivo

Gerar e avaliar um corpus sintético do AMIVE a partir de paráfrases automáticas, com o objetivo de:

- **Preservar a privacidade** dos autores das postagens originais
- **Manter a qualidade linguística** e semântica dos dados
- **Viabilizar a reprodutibilidade** de pesquisas que utilizem o AMIVE
- **Avaliar a qualidade** do corpus sintético via classificação de sintomas de saúde mental

---

## 🔄 Visão geral do pipeline

```
Texto original (PT)
        ↓
  Tradução PT → EN             (LLM via Groq / OpenAI / MariTalk)
        ↓
  Split automático             (textos > 60 tokens: divide por ponto e vírgula)
        ↓
  Paráfrase em EN              (Pegasus: tuner007/pegasus_paraphrase)
        ↓
  Tradução EN → PT             (LLM via Groq / OpenAI / MariTalk)
        ↓
  Seleção da melhor paráfrase  (BERTScore: neuralmind/bert-base-portuguese-cased)
        ↓
  CSV sintético final
```

Todas as etapas (incluindo o tratamento de textos longos e as correções) acontecem dentro do mesmo loop principal, sem scripts auxiliares separados.

---

## 📂 Estrutura do repositório

```
amive-synthetic/
│
├── README.md
├── README_en.md
├── requirements.txt
├── .env.example
│
├── pipeline/
│   └── main.py                   ← pipeline completo de geração
│
├── evaluation/
│   ├── classify.py               ← classificação binária (TF-IDF + múltiplos classificadores)
│   ├── intrinsic_evaluation.py   ← vocabulário, D-2, BLEU invertido, POS tags
│   ├── audit_stats.py            ← auditoria e estatísticas do processo de geração
│   ├── results_visualization.py  ← gráficos dos resultados de classificação
│   └── wordcloud_comparison.py   ← wordclouds original vs sintético
│
├── scripts/
│   ├── run.sh                    ← executa o pipeline principal
│   ├── run_classify.sh           ← executa a classificação
│   └── run_audit_stats.sh        ← executa a auditoria
│
└── utils/
    ├── io.py
    └── llm_request.py
```

---

## 🛠️ Tecnologias e dependências principais

- **Python 3.10**
- **[Pegasus](https://huggingface.co/tuner007/pegasus_paraphrase)** — geração de paráfrases em inglês
- **[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased)** — seleção da melhor paráfrase via BERTScore
-  **[Groq](https://console.groq.com)** / OpenAI / MariTalk — tradução via LLM
- **scikit-learn** — classificadores TF-IDF para avaliação extrínseca
- **[spaCy](https://spacy.io/)** (`pt_core_news_sm`) — análise de categorias gramaticais (POS tags)

> **Atenção:** as versões do `transformers` e `sentencepiece` são críticas para o funcionamento correto do Pegasus. Use exatamente as versões especificadas no `requirements.txt`.

---

## 🚀 Como executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/amive-synthetic.git
cd amive-synthetic
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe o modelo do spaCy

```bash
python -m spacy download pt_core_news_sm
```

### 5. Configure as chaves de API

```bash
cp .env.example .env
```

Preencha o `.env` com suas chaves:

```
GROQ_API_KEY=sua_chave_aqui
OPENAI_API_KEY=sua_chave_aqui      # opcional
MARITALK_API_KEY=sua_chave_aqui    # opcional
```

O Groq é recomendado por ser gratuito. Crie uma conta em [console.groq.com](https://console.groq.com).

---

## ▶️ Executando cada etapa

### Etapa 1: Geração do corpus sintético

```bash
chmod +x scripts/run.sh
./scripts/run.sh
```

Ou diretamente:

```bash
python pipeline/main.py \
  --dataset       dados/amive_sentences.csv \
  --doc_type      csv \
  --config        config/config.yaml \
  --model         llama-3.3-70b-versatile \
  --client        groq \
  --output        saida/amive_sintetico.csv \
  --output_stats  saida/amive_sintetico_stats.jsonl \
  --num_sequences 5 \
  --batch_size    5 \
  --sleep         1
```

O progresso é salvo a cada instância no arquivo `--output_stats`, permitindo retomar a execução em caso de interrupção sem reprocessar instâncias já concluídas.

**Argumentos principais:**

| Argumento | Descrição |
|---|---|
| `--dataset` | Arquivo de entrada (CSV, JSONL ou diretório TXT) |
| `--doc_type` | Formato: `csv`, `jsonl` ou `txt` (default: `txt`) |
| `--config` | YAML com configurações do LLM (`temperature`, `top_p`, `max_completion_tokens`) |
| `--model` | Nome do modelo LLM (ex: `llama-3.3-70b-versatile`) |
| `--client` | Cliente da API: `groq`, `openai` ou `maritalk` |
| `--output` | CSV de saída com as paráfrases finais |
| `--output_stats` | JSONL com estatísticas detalhadas por instância |
| `--num_sequences` | Número de paráfrases geradas por texto (default: `5`) |
| `--batch_size` | Instâncias por batch de tradução (default: `5`) |
| `--sleep` | Segundos entre requisições ao LLM (default: `1`) |

**Formato de entrada CSV:**
```
DOCNO,TEXT,SYMPTOM
100_1,"Estou me sentindo muito mal.",Tristeza/Humor depressivo
```

**Formato do YAML de configuração:**
```yaml
temperature: 0.3
top_p: 0.95
max_completion_tokens: 1024
```

---

### Etapa 2: Avaliação intrínseca

Calcula métricas de qualidade do corpus sintético comparado ao original:

| Métrica | Descrição |
|---|---|
| **Vocabulário** | Tamanho do vocabulário único |
| **D-2** | Proporção de bigramas únicos — quanto maior, mais diverso |
| **BLEU invertido** | Similaridade estrutural com o original — quanto menor, mais diverso |
| **Distribuição de POS** | Proporção de cada categoria gramatical via spaCy |

```bash
python evaluation/intrinsic_evaluation.py
```

> Edite as variáveis `ORIGINAL`, `SINTETICO`, `OUTPUT` e `OUTPUT_POS` no topo do arquivo antes de executar.

---

### Etapa 3: Classificação

Treina e avalia classificadores binários (sintoma vs. controle) com TF-IDF em quatro cenários:

| Cenário | Treino | Teste |
|---|---|---|
| 1 | Original | Original |
| 2 | Sintético | Sintético |
| 3 | Sintético | Original |
| 4 | Original | Sintético |

Além das métricas padrão (acurácia, precisão, recall, F1), calcula a similaridade de **Jaccard entre pares de classificadores** (indicando se os modelos acertam e erram nas mesmas instâncias).

```bash
chmod +x scripts/run_classify.sh
./scripts/run_classify.sh
```

Ou diretamente:

```bash
python evaluation/classify.py \
  --dataset                    dados/amive_sentences.csv \
  --dataset_sintetico          dados/amive_sintetico.csv \
  --dataset_controle           dados/amive_controle.csv \
  --dataset_controle_sintetico dados/amive_controle_sintetico.csv \
  --symptom                    "Tristeza/Humor depressivo" \
  --output                     results/results.csv \
  --output_jaccard             results/jaccard.csv \
  --test_size                  0.3 \
  --seed                       42
```

---

### Etapa 4: Auditoria do pipeline

Gera um relatório detalhado sobre o processo de geração, incluindo total de instâncias mantidas e excluídas, motivos de exclusão, estatísticas de BERTScore, diferença de vocabulário (Jaccard) e comparação de tamanho em tokens.

```bash
chmod +x scripts/run_audit_stats.sh
./scripts/run_audit_stats.sh
```

Ou diretamente:

```bash
python evaluation/audit_stats.py \
  --stats            saida/amive_sintetico_stats.jsonl \
  --dataset_original dados/amive_sentences.csv \
  --csv_final        saida/amive_sintetico.csv \
  --output           results/audit_report.txt \
  --output_csv       results/audit_detailed.csv
```

---

### Etapa 5: Visualizações

```bash
# gráficos dos resultados de classificação
python evaluation/results_visualization.py

# wordclouds original vs sintético
python evaluation/wordcloud_comparison.py
```

> Edite as variáveis de caminho no topo de cada arquivo antes de executar.

Arquivos gerados em `results/plots/`: tabela comparativa, gráfico de barras por métrica, barras por modelo, heatmaps de diferença e gráficos de distribuição de POS.

---

## ⚠️ Notas sobre limites de API

O Groq (plano gratuito) tem limite de **1000 requisições por dia**. O checkpoint automático garante que o progresso não se perde em caso de interrupção: basta rodar o mesmo comando novamente para retomar de onde parou.

Modelos recomendados:

| Modelo | Cliente | Característica |
|---|---|---|
| `llama-3.3-70b-versatile` | Groq | Melhor qualidade de tradução |
| `llama-3.1-8b-instant` | Groq | Mais rápido, menor consumo de cota |
| `sabia-3` | MariTalk | Boa alternativa para português brasileiro |

---

## 👩‍💻 Autoria e instituição

Trabalho desenvolvido como Trabalho de Conclusão de Curso (TCC) pela instituição UFSCar, curso Engenharia de Computação.

**Título:** Geração e Avaliação de Dados Sintéticos para Aplicações em Saúde Mental

**Aluna:** Vitória Rodrigues Pinto Borelli Figueiredo

**Orientadora:** Profa. Dra. Helena de Medeiros Caseli
