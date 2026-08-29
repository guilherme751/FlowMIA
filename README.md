# FlowMIA: Avaliando Ataques de Inferêcia de Membros em Modelos Generativos de Dados de Fluxo de Rede

Guilherme Silveira Gomes Brotto

FlowMIA é um framework para avaliar riscos de privacidade em conjuntos de dados sintéticos de fluxo de rede usando Ataques de Inferência de Pertencimento (MIAs) baseados em modelos generativos profundos.

Ele fornece um pipeline de avaliação sistemático que abrange:

- Ataques de inferência de pertencimento usando um atacante baseado em GAN (FlowMIA-GAN)
- Ataque de densidade DOMIAS, baseado em normalizing flows (BNAF)
- Análise de privacidade baseada em distância (Distance to Closest Record – DCR)
- Avaliação de fidelidade estatística
- Avaliação de utilidade de aprendizado de máquina
- Análise de trade-off entre privacidade, fidelidade e utilidade

FlowMIA foi projetado especificamente para dados de fluxo de rede, mas pode ser aplicado a qualquer conjunto de dados tabulares. Novos ataques de inferência de pertencimento podem ser adicionados sem modificar o código principal (veja [Estendendo o FlowMIA com um novo ataque](#estendendo-o-flowmia-com-um-novo-ataque)).

## Visão Geral

Dados sintéticos são cada vez mais usados para compartilhar tráfego de rede sensível preservando a privacidade. No entanto, modelos generativos podem memorizar amostras de treinamento, expondo informações privadas.

FlowMIA simula um adversário realista que tem acesso a:

- Dados sintéticos gerados por um modelo alvo
- Dados de referência (não-membros)
- Conhecimento da estrutura dos dados

O atacante treina uma GAN, chamada de FlowMIA-GAN, para realizar um ataque de inferência de pertencimento, estimando se amostras específicas fizeram parte do conjunto de treinamento.

Por que usar um dataset de referência como não-membro em vez do conjunto de teste? Dados de fluxo de rede possuem, por sua natureza, muitas linhas praticamente idênticas — usar o conjunto de teste (amostrado da mesma distribuição do treino) como não-membro tende a subestimar o vazamento de privacidade. Um dataset de referência de domínio similar, mas de origem distinta, é um não-membro mais realista. Ainda assim, quem quiser usar o conjunto de teste como não-membro pode fazê-lo — basta apontar `non_member_path` para ele.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/guilherme751/FlowMIA.git
cd FlowMIA
```

Instale o pacote (modo editável, recomendado durante desenvolvimento):

```bash
pip install -e .            # dependências principais (CPU)
pip install -e ".[gpu]"     # + stack CUDA/torch para treinar em GPU
pip install -e ".[dev]"     # + pytest, para rodar a suíte de testes
pip install -e ".[notebooks]"  # + jupyter/ipykernel/plotly, para rodar examples/
```

## Estrutura do Projeto

```
FlowMIA/
│
├── src/flowmia/            # pacote instalável
│   ├── core.py             # classe FlowMIA
│   ├── config.py           # FlowMIAConfig + load_config() (YAML)
│   ├── fidelity.py         # métricas de fidelidade
│   ├── utility.py          # métricas de utilidade (RTR/TSTR)
│   └── attacks/             # FlowMIA-GAN, DOMIAS, DCR + registro de ataques
│
├── datasets/
│   ├── real/               # conjuntos de dados membros (dados de treinamento)
│   ├── reference/          # conjuntos de dados não-membros
│   └── synthetic/          # conjuntos de dados sintéticos gerados por modelos
│
├── examples/
│   ├── configs/             # exemplos de arquivo de configuração YAML
│   ├── exemplo_pratico.ipynb  # exemplo de uso
│   └── teste.ipynb
│
├── tests/                  # suíte de testes (pytest)
│
├── pyproject.toml
└── README.md
```

## Conceitos Principais

FlowMIA usa três tipos de conjuntos de dados obrigatórios, mais um opcional:

| Conjunto de Dados | Descrição |
|-------------------|-----------|
| Membro | Dados reais usados para treinar o modelo generativo alvo |
| Não-membro | Conjunto de dados de referência não usado durante o treinamento |
| Sintético | Dados gerados pelo modelo alvo |
| Teste (opcional) | Amostra da mesma distribuição dos membros, usada apenas para avaliação de utilidade |

O atacante tem acesso a:

- Dados sintéticos
- Dados de referência
- Nenhum acesso aos rótulos reais de treinamento

## Início Rápido

### Passo 1 — Importar FlowMIA

```python
from flowmia import FlowMIA
```

### Passo 2 — Criar Configuração

Via dict Python:

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
    'use_wgan': True, # se deve usar WGAN ou GAN tradicional
    'batch_size': 200, # número de amostrar por lote
    'num_epochs': 10, # número de épocas
    'fcheckpoint': 5, # frequência para salvar o checkpoint
    'save_path': 'meus_resultados/netshare'    # pasta para salvar resultados
}
```

Ou via arquivo de configuração YAML (mesmas chaves do dict acima — veja
`examples/configs/netshare_example.yaml`):

```yaml
member_path: datasets/real/cidds_train.csv
non_member_path: datasets/reference/ton.csv
synth_path: datasets/synthetic/netshare.csv
test_path: datasets/real/cidds_test.csv
save_path: meus_resultados/netshare

categorical_cols: [proto]
numerical_cols: [srcport, dstport, td, pkt, byt]
ip_cols: [srcip, dstip]
label_col: label

use_wgan: true
batch_size: 200
num_epochs: 10
fcheckpoint: 5
```

### Passo 3 — Inicializar FlowMIA

```python
flowmia = FlowMIA(config=config)

# ou, a partir de um arquivo YAML:
flowmia = FlowMIA.from_yaml("examples/configs/netshare_example.yaml")
```

### Passo 4 — Executar Ataque de Inferência de Pertencimento

```python
mia_results = flowmia.flowmiagan(plot=True)
```

Isso treina a GAN do atacante e avalia a performance da inferência de pertencimento.

As saídas incluem:

- AUC
- Acurácia, precisão, recall, F1
- Scores de predição para membros, não-membros, sintéticos e ruído aleatório

### Passo 5 — Calcular Distance to Closest Record (DCR)

```python
scores, auc = flowmia.compute_dcr(test_size=5000)
```

DCR mede a distância mínima entre amostras de teste e amostras sintéticas.

DCR mais baixo (score mais alto) indica maior risco de memorização.

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

classifiers = [
    MLPClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
]

utility_results = flowmia.evaluate_utility(
    classifiers=classifiers,
    plot=True
)
```

Isso avalia quão úteis os dados sintéticos são para treinar modelos de ML.

## Exemplo Completo

```python
from flowmia import FlowMIA
from sklearn.tree import DecisionTreeClassifier

config = {
    'member_path': 'datasets/real/cidds_train.csv', # path dos membros
    'non_member_path': 'datasets/reference/ton.csv', # path dos não-membros
    'synth_path': 'datasets/synthetic/netshare.csv', # path dos sintéticos
    'test_path': 'datasets/real/cidds_test.csv', # path do teste
    'categorical_cols': ['proto'], # colunas categóricas
    'numerical_cols': ['srcport', 'dstport', 'td', 'pkt', 'byt'], #colunas numéricas
    'ip_cols': ['srcip', 'dstip'], # colunas de ip
    'label_col': 'label', # nome da coluna do rótulo 
    'use_wgan': True, # se deve usar WGAN ou GAN tradicional
    'batch_size': 200, # número de amostrar por lote
    'num_epochs': 500, # número de épocas
    'fcheckpoint': 100, # frequência para salvar o checkpoint
    'save_path': 'meus_resultados/netshare'    # pasta para salvar resultados
}

flowmia = FlowMIA(config)

mia_results = flowmia.flowmiagan()

scores, auc = flowmia.compute_dcr()

fidelity_results = flowmia.evaluate_fidelity()

utility_results = flowmia.evaluate_utility([DecisionTreeClassifier()])
```

## Modelo de Ataque FlowMIA

FlowMIA usa um atacante baseado em GAN:

- **Gerador** aprende a modelar a distribuição de dados sintéticos
- **Discriminador** aprende a distinguir entre amostras membros e não-membros

A saída do discriminador é usada como o score de inferência de pertencimento.

O pré-processamento é ajustado usando:

- Dados sintéticos
- Dados membros

Isso garante conhecimento realista do atacante mantendo o escalonamento adequado de features.

## Estendendo o FlowMIA com um novo ataque

Todo ataque de inferência de pertencimento implementa a interface `BaseAttack`
(`fit()` + `attack()`) e se registra com o decorator `@register_attack`:

```python
from flowmia.attacks import AttackResult, BaseAttack, register_attack

@register_attack("meu_ataque")
class MeuAtaque(BaseAttack):
    def fit(self, X_member, X_non_member, X_synth, **kwargs):
        # treina/prepara o ataque
        return self

    def attack(self, X_member, X_non_member, test_size=1000, **kwargs):
        # calcula os scores de pertencimento
        return AttackResult(scores=..., auc=...)
```

Uma vez registrado, o novo ataque pode ser executado como qualquer outro:

```python
resultados = flowmia.evaluate_privacy(attacks=["meu_ataque"])
```

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
