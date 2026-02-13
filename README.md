# FlowMIA: Avaliando Ataques de Inferêcia de Membros em Modelos Generativos de Dados de Fluxo de Rede

Guilherme Silveira Gomes Brotto

Segurança em Computação

Vídeo apresentação: [Vídeo](https://youtu.be/E95B0ZUr53w)
Slides apresentação: [Slides](https://docs.google.com/presentation/d/1JTfF8BNcnTJOkX6QXxeCDJtobfV9eWn0i6LI9AD82f4/edit?usp=sharing)

FlowMIA é um framework para avaliar riscos de privacidade em conjuntos de dados sintéticos de fluxo de rede usando Ataques de Inferência de Pertencimento (MIAs) baseados em modelos generativos profundos.

Ele fornece um pipeline de avaliação sistemático que abrange:

- Ataques de inferência de pertencimento usando um atacante baseado em GAN
- Análise de privacidade baseada em distância (Distance to Closest Record – DCR)
- Avaliação de fidelidade estatística
- Avaliação de utilidade de aprendizado de máquina
- Análise de trade-off entre privacidade, fidelidade e utilidade

FlowMIA foi projetado especificamente para dados de fluxo de rede, mas pode ser aplicado a qualquer conjunto de dados tabulares.

## Visão Geral

Dados sintéticos são cada vez mais usados para compartilhar tráfego de rede sensível preservando a privacidade. No entanto, modelos generativos podem memorizar amostras de treinamento, expondo informações privadas.

FlowMIA simula um adversário realista que tem acesso a:

- Dados sintéticos gerados por um modelo alvo
- Dados de referência (não-membros)
- Conhecimento da estrutura dos dados

O atacante treina uma GAN, chamada de FlowMIA-GAN, para realizar um ataque de inferência de pertencimento, estimando se amostras específicas fizeram parte do conjunto de treinamento.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/guilherme751/FlowMIA.git
cd FlowMIA
```

Instale as dependências:

```bash
pip install -r requirements.txt
```


## Estrutura do Projeto

```
flowmia/
│
├── src/
│   ├── fidelity/ # Código de fidelidade              
│   ├── utility/ # Código de utilidade       
│   └── privacy/ # Código do FlowMIA-GAN
│
├── datasets/
│   ├── real/               # Conjuntos de dados membros (dados de treinamento)
│   ├── reference/          # Conjuntos de dados não-membros
│   └── synthetic/          # Conjuntos de dados sintéticos gerados por modelos
│
├── exemplo_pratico.ipynb # Exemplo de uso
│   
│
└── README.md
```

## Conceitos Principais

FlowMIA usa três tipos de conjuntos de dados:

| Conjunto de Dados | Descrição |
|-------------------|-----------|
| Membro | Dados reais usados para treinar o modelo generativo alvo |
| Não-membro | Conjunto de dados de referência não usado durante o treinamento |
| Sintético | Dados gerados pelo modelo alvo |

O atacante tem acesso a:

- Dados sintéticos
- Dados de referência
- Nenhum acesso aos rótulos reais de treinamento

## Início Rápido

### Passo 1 — Importar FlowMIA

```python
from src.flowmia import FlowMIA
```

### Passo 2 — Criar Configuração

```python
config = {
    'member_path': 'datasets/real/cidds_train.csv', # path dos membros
    'non_member_path': 'datasets/reference/ton.csv', # path dos não-membros
    'synth_path': 'datasets/synthetic/netshare.csv', # path dos sintéticos
    'test_path': 'datasets/real/cidds_test.csv', # path do teste
    'categorical_cols': ['proto'], # colunas categóricas
    'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'], #colunas numéricas
    'ip_cols': ['srcip', 'dstip'], # colunas de ip
    'label_col': 'label', # nome da coluna do rótulo 
    'batch_size': 200, # número de amostrar por lote
    'num_epochs': 10, # número de épocas
    'fcheckpoint': 5, # frequência para salvar o checkpoint
    'save_path': 'teste/netshare'    # pasta para salvar resultados
}
```

### Passo 3 — Inicializar FlowMIA

```python
flowmia = FlowMIA(config=config)
```

### Passo 4 — Executar Ataque de Inferência de Pertencimento

```python
history, mia_results = flowmia.flowmiagan(plot=True)
```

Isso treina a GAN do atacante e avalia a performance da inferência de pertencimento.

As saídas incluem:

- AUC
- Acurácia
- Curvas de perda do ataque
- Scores de predição

### Passo 5 — Calcular Distance to Closest Record (DCR)

```python
dcr_results = flowmia.compute_dcr(n_sample=5000)
```

DCR mede a distância mínima entre amostras sintéticas e amostras reais.

DCR mais baixo indica maior risco de memorização.

### Passo 6 — Avaliar Fidelidade Estatística

```python
fidelity_results = flowmia.evaluate_fidelity(plot=True)
```

As métricas incluem:

- Divergência KL
- Divergência JS
- Distância Wasserstein

Essas métricas medem a similaridade entre distribuições reais e sintéticas.

### Passo 7 — Avaliar Utilidade

```python
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifiers = {
    "MLP": MLPClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

utility_results = flowmia.evaluate_utility(
    classifiers=classifiers,
    plot=True
)
```

Isso avalia quão úteis os dados sintéticos são para treinar modelos de ML.

## Exemplo Completo

```python
{
    'member_path': 'datasets/real/cidds_train.csv', # path dos membros
    'non_member_path': 'datasets/reference/ton.csv', # path dos não-membros
    'synth_path': 'datasets/synthetic/netshare.csv', # path dos sintéticos
    'test_path': 'datasets/real/cidds_test.csv', # path do teste
    'categorical_cols': ['proto'], # colunas categóricas
    'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'], #colunas numéricas
    'ip_cols': ['srcip', 'dstip'], # colunas de ip
    'label_col': 'label', # nome da coluna do rótulo 
    'batch_size': 200, # número de amostrar por lote
    'num_epochs': 500, # número de épocas
    'fcheckpoint': 100, # frequência para salvar o checkpoint
    'save_path': 'teste/netshare'    # pasta para salvar resultados
}

flowmia = FlowMIA(config)

history, mia_results = flowmia.flowmiagan()

dcr_results = flowmia.compute_dcr()

fidelity_results = flowmia.evaluate_fidelity()

utility_results = flowmia.evaluate_utility(classifiers)
```

## Modelo de Ataque FlowMIA

FlowMIA usa um atacante baseado em GAN:

- **Gerador** aprende a modelar a distribuição de dados sintéticos
- **Discriminador** aprende a distinguir entre amostras membros e não-membros

A saída do discriminador é usada como o score de inferência de pertencimento.

O pré-processamento é ajustado usando:

- Dados sintéticos
- Dados não-membros

Isso garante conhecimento realista do atacante mantendo o escalonamento adequado de features.

## Métricas de Avaliação

### Métricas de Privacidade

**Ataque de Inferência de Pertencimento:**

- AUC
- Acurácia, Precisão, Recall, F1
- Indicadores de vazamento de privacidade, como gaps médios

**Distance to Closest Record (DCR):**

- Mede risco de memorização
- Detecta cópias exatas ou quase exatas

### Métricas de Fidelidade

Mede similaridade estatística:

- Divergência KL
- Divergência JS
- Distância Wasserstein

### Métricas de Utilidade

Mede a utilidade dos dados sintéticos:

- Acurácia de classificação
- Comparação de performance de modelos

## Modelos Generativos Suportados

FlowMIA pode avaliar dados sintéticos gerados por qualquer modelo:

- CTGAN
- NetShare
- Tabula
- TVAE
- Modelos de difusão
- Geradores personalizados

FlowMIA é agnóstico ao modelo.

## Contexto de Pesquisa

FlowMIA foi desenvolvido para avaliar riscos de privacidade na geração de dados sintéticos de fluxo de rede.

Ele aborda o trade-off fundamental entre:

- Privacidade
- Fidelidade
- Utilidade


## Autor

**Guilherme S. G. Brotto**  
Universidade Federal do Espírito Santo (UFES)  
Laboratório de Pesquisa em Redes e Multimídia (LPRM)
