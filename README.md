## Estrutura e Histórico dos Modelos

O arquivo [`model.py`](./model.py) foi desenvolvido inicialmente para testes e validação das primeiras abordagens do desafio. Posteriormente, o código foi adaptado e aprimorado no notebook [`mode.ipynb`](./mode.ipynb), que foi entregue como solução final em formato de notebook.

Os notebooks [`modelo_startup.ipynb`](./modelo_startup.ipynb) e [`projeto_kaggle.ipynb`](./projeto_kaggle.ipynb) representam outras tentativas de modelagem, mas não apresentaram desempenho satisfatório para serem considerados como solução final.

As submissões estão relacionadas da seguinte forma:
- [`submission.csv`](./submission.csv): Resultado gerado a partir do notebook [`mode.ipynb`](./mode.ipynb).
- [`submission_1.csv`](./submission_1.csv): Resultado gerado a partir do notebook [`projeto_kaggle.ipynb`](./projeto_kaggle.ipynb).

# Desafio de Previsão de Sucesso de Startups
Imagine que você foi contratado por uma das maiores aceleradoras do mundo. Sua missão? Desenvolver um modelo preditivo capaz de identificar quais startups apresentam maior probabilidade de se tornarem casos de sucesso no mercado.

A aceleradora, que já lançou unicórnios e empresas líderes globais, busca otimizar seus investimentos e estratégias de aceleração, apostando nas startups certas e maximizando o impacto econômico.

Você receberá uma base de dados com centenas de startups, incluindo informações sobre:
- 📈 Histórico de captação de recursos
- 🌍 Localização
- 🏭 Setor de atuação
- 🔗 Conexões estratégicas
- 🏆 Marcos alcançados
Seu desafio será usar esses dados para prever se uma startup terá sucesso ou não.

## Objetivo
Criar um modelo de machine learning que preveja, com boa acurácia, se uma startup será bem-sucedida.
Essa previsão apoiará investidores e aceleradoras na tomada de decisões mais estratégicas.

## Dados Disponíveis
A base contém informações de startups em arquivos separados para treino e teste:

train.csv → conjunto de treinamento com startups + variável alvo (labels)
test.csv → conjunto de teste (sem a coluna alvo)
sample_submission.csv → modelo de submissão esperado

Este conjunto de dados reúne informações reais sobre startups de diferentes setores, incluindo histórico de rodadas de investimento, valores captados, localização e áreas de atuação.
O objetivo é prever se uma startup terá sucesso (ativa/adquirida) ou insucesso (fechada) com base nessas variáveis.

A base foi adaptada para fins acadêmicos: identificadores, colunas que poderiam gerar vazamento e valores inconsistentes foram removidos. Alguns campos podem conter valores ausentes (NaN), refletindo casos em que o evento não ocorreu ou não foi registrado.

Mais do que buscar o melhor desempenho, este desafio incentiva os participantes a explorar técnicas de pré-processamento, seleção de variáveis e modelagem preditiva aplicadas ao empreendedorismo e inovação.

## Entregáveis
- Notebook Completo: Um notebook Jupyter documentando todo o processo, desde a exploração dos dados até a criação e avaliação do modelo. Você pode trabalhar com o notebook dentro da plataforma Kaggle ou importar um arquivo .ipynb
- Arquivo CSV de resultados: Submeta os resultados em csv do seu melhor modelo treinado, conforme template disponibilizado.

## Regras do Campeonato
- Utilize seu e-mail do Inteli na competição, para que a gente possa identificar você e sua entrega.
- Sua participação deve ser individual!
- Utilize Python e apenas as bibliotecas padrão do módulo: Numpy, Pandas, ScikitLearn.
- Encorajamos o uso de bibliotecas de visualização e gráficos para fortalecer suas análises e justificar suas escolhas. Para isso, utilize bibliotecas como: Matplotlib, Seaborn e/ou Plotly.
- Não é permitido utilizar outras bibliotecas! Caso queira implementar algoritmos mais avançados, deverá fazê-lo apenas com as ferramentas permitidas.
- Não é permitido usar dados externos além do fornecido.
- O ranqueamento será dado conforme a performance do seu modelo na métrica de acurácia. Quanto maior, melhor! Ao final, a pessoa que ficar em primeiro lugar na turma ganhará um prêmio (surpresa!). Haverá também prêmio exclusivo para a melhor acurácia dentre todas as turmas de primeiro ano! (critérios de desempate: outras métricas como precisão e recall, além da nota final da entrega)
- Não trapaceie! Se seu código possuir semelhança a alguma outra solução pronta, você será desclassificado(a) e ficará com nota zero! (sujeito a sanções disciplinares previstas no regulamento do Inteli)

## Critérios de Avaliação das Submissões

- Limpeza e Tratamento de Valores Nulos (até 0,5 pt):
    - A qualidade dos dados é crucial. Demonstre seu processo de limpeza, incluindo a maneira como lida com valores ausentes e outliers que possam distorcer os resultados.
- Codificação de Variáveis Categóricas (até 0,5 pt):
    - Aplique técnicas apropriadas de codificação para transformar variáveis categóricas em formatos utilizáveis em modelos preditivos, garantindo que a informação essencial não seja perdida no processo.
- Exploração e Visualização dos Dados (até 2,0 pts):
    - Realize uma análise exploratória detalhada para descobrir padrões, correlações e tendências nos dados. Use visualizações eficazes para comunicar seus insights e justificar suas escolhas de features e modelos.
- Formulação de Hipóteses (até 1,0 pt):
    - Formule três hipóteses que possam explicar os fatores que influenciam o sucesso da empresas. Por exemplo, pode-se investigar se a empresas com mais funcionários ou com menos tempo de fundação têm maior chance de sucesso.
- Seleção de Features (até 1,0 pt):
    - Escolha as features mais relevantes para o modelo com base em sua análise exploratória e hipóteses formuladas.
- Construção e Avaliação do Modelo (até 2,0 pts):
    - Selecione um modelo de machine learning adequado (ou uma combinação de modelos) que maximize a capacidade preditiva. A avaliação deve incluir métricas como acurácia, precisão, recall, e F1-score.
- Finetuning de Hiperparâmetros (até 1,0 pt):
    - Realize um ajuste fino (finetuning) dos hiperparâmetros do modelo para otimizar seu desempenho. Detalhe o processo de busca e as justificativas para as escolhas feitas.
- Acurácia Mínima (até 2,0 pts):
    - O modelo deve atingir uma acurácia mínima de 80% para ser considerado bem-sucedido (pontuação total). Embora a acurácia seja a métrica principal usada na competição, analise também outras métricas como precisão e recall, para melhor interpretação do desempenho do modelo preditivo treinado.
- Documentação e Apresentação dos Resultados (demérito de até 2,0 pts):
    - A documentação clara e a apresentação dos resultados são importantes. O notebook final deve ser bem organizado, com código limpo, e o raciocínio por trás de cada decisão deve ser explicado de forma objetiva e compreensível em células de texto, sem exageros.