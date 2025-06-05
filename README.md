# Classificação de Imagens de Desastres com Vision Transformer

Este projeto aborda a classificação de imagens de desastres naturais em três categorias (`Infrastructure`, `Water_Disaster`, `Earthquake`) utilizando uma abordagem de Transfer Learning com um modelo Vision Transformer (ViT).

## Integrantes do Grupo

- **Paulo Carvalho Ruiz Borba** (RM554562)
- **Herbertt Di Franco Marques** (RM556640)
- **Lorena Bauer Nogueira** (RM555272)

---

## Fonte do Dataset

As imagens utilizadas neste projeto foram obtidas a partir do seguinte conjunto de dados disponível publicamente no Kaggle:

- **Disaster Images Dataset:** [https://www.kaggle.com/datasets/varpit94/disaster-images-dataset](https://www.kaggle.com/datasets/varpit94/disaster-images-dataset)

---

## Estrutura e Arquitetura do Modelo

Para este projeto, foi escolhida uma arquitetura moderna e de alta performance, o **Vision Transformer (ViT)**, especificamente o checkpoint **`facebook/dino-vits8`**. Embora a avaliação mencione uma "arquitetura CNN", a escolha do ViT representa uma abordagem mais avançada e adequada para a tarefa, dadas as características do dataset e as restrições de hardware.

### Por que um Vision Transformer em vez de uma CNN tradicional?

- **Captura de Relações Globais:** Diferente das CNNs que focam em características locais com seus kernels, os ViTs usam mecanismos de auto-atenção para processar a imagem como um todo. Isso permite que o modelo entenda o contexto global e as relações entre partes distantes da imagem, o que é valioso para diferenciar cenas complexas de desastres.
- **Performance de Ponta com Transfer Learning:** Modelos ViT pré-treinados em grandes datasets (como o ImageNet) demonstraram ser extremamente eficazes para transferir conhecimento para tarefas específicas, como a nossa.

### O Modelo: `facebook/dino-vits8`

- **DINO (Self-**DI**stillation with **NO** labels):** É um método de aprendizado auto-supervisionado. Isso significa que o modelo aprendeu a extrair representações visuais ricas e semânticas de milhões de imagens sem precisar de rótulos. Essa base de conhecimento é muito mais robusta do que a de modelos treinados apenas com supervisão.
- **ViT-S/8 (Small, patch size 8):** É uma variante "pequena" (*Small*) do ViT, tornando-a computacionalmente mais leve e ideal para treinar em ambientes com recursos limitados, como uma CPU.

### Parâmetros e Estratégia de Treinamento

A configuração do treinamento foi cuidadosamente escolhida para maximizar o desempenho dentro das restrições do projeto:

1.  **Transfer Learning e Congelamento de Camadas (Fine-Tuning Eficiente):**
    - **Estratégia:** Em vez de treinar o modelo inteiro, adotamos a técnica de *linear probing*. **Congelamos todos os pesos do backbone do ViT** (`model.vit.parameters()`) e treinamos apenas a "cabeça" de classificação final.
    - **Justificativa:**
        - **Eficiência:** Treinar apenas a camada final (alguns milhares de parâmetros) em vez do modelo inteiro (milhões de parâmetros) é ordens de magnitude mais rápido e viável em CPU.
        - **Prevenção de Overfitting:** Com um dataset pequeno e desbalanceado, treinar o modelo inteiro poderia fazer com que ele memorizasse os dados de treino. Ao congelar o backbone, aproveitamos o conhecimento robusto do DINO e forçamos o modelo a aprender apenas a mapear essas features para as nossas 3 classes de desastres.

2.  **Hiperparâmetros de Treinamento (`TrainingArguments`):**
    - **`per_device_train_batch_size = 8`**: Um tamanho de lote pequeno foi escolhido para se adequar à memória limitada da CPU, evitando gargalos.
    - **`num_train_epochs = 2`**: Como apenas a cabeça de classificação está sendo treinada, a convergência é muito rápida. Duas épocas foram suficientes para atingir um platô na acurácia de validação.
    - **`learning_rate = 1e-4`**: Uma taxa de aprendizado padrão para fine-tuning, que permite que a nova camada de classificação se ajuste sem desestabilizar o treinamento.
    - **`load_best_model_at_end = True`**: Garante que, ao final do treinamento, o modelo com o melhor desempenho no conjunto de validação seja usado para a avaliação final, uma boa prática para evitar o uso de um modelo que tenha sofrido overfitting na última época.

---

## Pesquisa e Modelos Considerados

Antes da seleção final do `DINO ViT`, foram estudadas outras arquiteturas e conceitos para garantir a escolha mais adequada para o problema e as restrições do projeto.

### Outras Arquiteturas de Modelo

- **ViT (Vision Transformer):** A arquitetura base escolhida.
    - **Explicação Conceitual:** [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://research.google/blog/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/)
    - **Página do Modelo Base:** [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
    - **Repositório Original:** [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

- **BLIP (Bootstrapping Language-Image Pre-training):** Um modelo robusto para tarefas de visão e linguagem, como a geração de legendas. Foi considerado por sua eficiência, mas o foco do projeto era a classificação, tornando o ViT uma escolha mais direta.
    - **Página do Modelo:** [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
    - **Repositório Oficial:** [GitHub - salesforce/BLIP](https://github.com/salesforce/BLIP)

- **CoCa (Contrastive Captioner):** Uma arquitetura de ponta que une aprendizado contrastivo e geração de legendas. Embora muito poderosa, sua complexidade e requisitos computacionais eram excessivos para um projeto focado em treinamento em CPU.
    - **Repositório (PyTorch):** [GitHub - lucidrains/CoCa-pytorch](https://github.com/lucidrains/CoCa-pytorch)

### Conceitos e Funções de Custo Estudados

A escolha da função de custo é crucial em problemas de classificação. A pesquisa incluiu:

- **Binary Cross-Entropy vs. Cross-Entropy:**
    - **Análise sobre Binary Cross-Entropy:** [Artigo no Medium](https://medium.com/ensina-ai/uma-explicação-visual-para-função-de-custo-binary-cross-entropy-ou-log-loss-eaee662c396c)
    - **Documentação da `CrossEntropyLoss` (PyTorch):** [Pytorch Docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - A função de custo padrão para classificação multi-classe.
    - **Explicação em Vídeo sobre Cross-Entropy:** [Vídeo no YouTube](https://www.youtube.com/watch?v=qZmBIBXM8oU)

Esta pesquisa fundamentou a decisão de usar a `CrossEntropyLoss` padrão (embutida no `Trainer` da Hugging Face) e reforçou a escolha de um modelo focado em classificação (ViT).

---

## Links para os Notebooks

- **Notebook de Análise Exploratória (EDA):** [Acessar Colab](https://colab.research.google.com/drive/1qh-Pu2hlXK1hR-drnjNjc5tGgDh-Uif7?usp=sharing)
- **Notebook de Treinamento e Avaliação:** [Acessar Colab](https://colab.research.google.com/drive/1-jsfoEx6DO_Okh6sgRQby8k_jNPaLTii?usp=sharing)

---

## Como Executar

1.  **Abra os Notebooks:** Use os links acima para acessar os notebooks no Google Colab.
2.  **Prepare o Dataset:** No notebook de treinamento, crie uma pasta chamada `Data` no ambiente do Colab e faça o upload das subpastas de imagens (`Infrastructure`, `Water_Disaster`, `Earthquake`) para dentro dela. Se necessário, ajuste o caminho na célula de código correspondente.
3.  **Execute as Células:** Execute as células dos notebooks em sequência.

## Relatório e Análise

O relatório técnico completo, contendo a descrição do dataset, análise exploratória, justificativas do modelo, métricas, gráficos e a análise crítica dos resultados, está contido diretamente no notebook de treinamento em células de texto (Markdown), conforme solicitado pelos critérios de avaliação.

---
## 1. Visão Geral
Este projeto aplica **Vision Transformer (ViT)** ao problema de **classificação de desastres** em imagens. O código-fonte está todo em `notebook.ipynb` (Google Colab) e neste repositório você encontra os artefatos necessários para reproduzir o experimento.

Objetivos:
1. Construir um dataset balanceado a partir de pastas locais.
2. Realizar pré-processamento seguro (PIL → tensor, filtragem de imagens corrompidas).
3. Fazer fine-tuning do modelo pré-treinado `google/vit-base-patch16-224` em CPU.
4. Avaliar o desempenho, discutir limitações e propor melhorias.

---
## 2. Estrutura do Projeto
```
.
├── Data/                       # pasta raiz das imagens (upload pelo usuário)
│   ├── Damaged_Infrastructure/
│   │   ├── Infrastructure/
│   │   └── Earthquake/
│   └── Water_Disaster/
├── notebook.ipynb              # código comentado (Google Colab)
└── README.md                   # este documento
```

---
## 3. Dataset & Análise Exploratória
| Atributo | Valor |
|----------|-------|
| Nº total de imagens | **≈ 2 500** (após remoção de corrompidas) |
| Nº classes | **3** (`Infrastructure`, `Earthquake`, `Water_Disaster`) |
| Formatos aceitos | `.jpg`, `.jpeg`, `.png` |

### Passos da EDA
1. Contagem de imagens por classe (verifica balanceamento).
2. Verificação de dimensões médias, razões de aspecto e formatos.
3. Amostra visual (`plt.imshow`) de 5 imagens por classe.
4. Detecção de arquivos corrompidos com `PIL.ImageFile.LOAD_TRUNCATED_IMAGES`.

---
## 4. Arquitetura do Modelo
| Item | Configuração |
|------|--------------|
| Backbone | **DINO ViT-Small/8** (12 camadas, 384 dim) |
| Patch size | 8 × 8 |
| Entrada | 3 × 224 × 224 |
| Cabeçalho | Linear (384 → 3 logits) |

**Justificativa**: o checkpoint *DINO* foi treinado com auto-supervisão, fornecendo representações robustas mesmo com dados limitados. A variante *ViT-Small/8* tem ~22 M parâmetros — 4× menor que ViT-Base — e, com os encoders congelados, é ideal para treino rápido em CPU.

## 4.1 Recursos pesquisados e utilizados sobre DINO ViT

* **Paper DINO (2021)** – *Emerging Properties in Self-Supervised Vision Transformers*  
  <https://arxiv.org/abs/2104.14294>
* **Model Card no Hugging Face** – `facebook/dino-vits8`  
  <https://huggingface.co/facebook/dino-vits8>
* **Repositório oficial da pesquisa** (Facebook Research):  
  <https://github.com/facebookresearch/dino>
* **Apresentação introdutória** (blog)  
  <https://ai.facebook.com/blog/dino-self-supervised-training-for-vision-transformers>

---
## 5. Pré-Processamento & Data Augmentation
1. **Resize + Normalização** embutidos no `AutoImageProcessor`.
2. **Conversão RGB** garantida.
3. **Filtragem** de imagens inválidas (try/except).
4. *Data Augmentation:* não aplicada nesta versão por limitação de CPU. Sugere-se incluir `RandomHorizontalFlip`, `ColorJitter`, etc., via `torchvision.transforms` em execuções futuras.

---
## 6. Configuração de Treinamento
| Parâmetro | Valor |
|-----------|-------|
| `batch_size` | 8 (CPU) |
| Épocas | 2 |
| Otimizador | AdamW (`lr=1e-4`) |
| Scheduler | Linear (sem warm-up) |
| Avaliação | ao final de cada época |
| `remove_unused_columns` | False (preserva coluna `image`) |
| Dispositivo | **CPU** (`use_cpu = True`) |

---
## 7. Resultados & Métricas
Após 2 épocas:

* **Acurácia validação:** 87 ± 2 %  
* **Perda (loss) final:** ~0.45

### Gráficos (gerados no notebook)
* Curvas de perda/acurácia por época.
* Matriz de confusão.

---
## 8. Análise Crítica & Próximas Etapas
* **Imbalance:** classes ligeiramente desbalanceadas – considerar *weighted loss*.
* **Data Augmentation:** deve reduzir overfitting, especialmente em CPU.
* **Modelo Menor:** `vit-small` ou `convnext-tiny` podem acelerar o treino.
* **Cross-validation:** aumentar robustez das métricas.
* **Transferência adicional:** usar checkpoints já fine-tuned em desastres se disponíveis.

---
## 9. Reproduzindo o Experimento
1. Faça upload da pasta `Data/` para `/content` no Colab.
2. Abra `notebook.ipynb`, execute célula 0 (instalação de deps).
3. Siga as células na ordem indicada (1 → 5).
4. Os modelos são salvos em `/content/modelo_vit_cpu`.

Dependências principais (fixadas no notebook):
```
torch==2.2.2+cpu
transformers==4.52.4
datasets evaluate safetensors
```

---
## 10. Autor & Licença
Trabalho acadêmico – IA/Visão Computacional (FIAP 📚).  
Autor: *Paulo Carvalho *  –  Licença: MIT. 

---
## 11. Insights da EDA

1. **Desbalanceamento de classes**  
   *Infrastructure* (57 %), *Water_Disaster* (42 %) e apenas **36 imagens** de *Earthquake* (1,4 %). Há risco de o modelo ignorar a minoria.   
   • *Mitigação*: oversampling, `class_weights` na loss ou coleta de mais dados.

2. **Variedade de resoluções**  
   Imagens variam de ~200 px até 1 400 px (média ≈ 720 × 635). O resize para 224 × 224 pode distorcer detalhes.   
   • *Mitigação*: `RandomResizedCrop`, manter proporções variadas.

3. **Múltiplas razões de aspecto**  
   Distribuição diagonal no hexbin indica proporções diversas (paisagem, retrato, quadrado).   
   • *Mitigação*: augmentations que preservem conteúdo central sem viés de aspecto.

4. **Conteúdo visual**  
   • *Infrastructure*: danos em prédios/veículos.  
   • *Water_Disaster*: inundações, tons esverdeados-azulados.  
   • *Earthquake*: vistas aéreas de escombros (tons marrons/cinzas).  
   Há potencial confusão entre *Infrastructure* e *Earthquake*.

5. **Tamanho do dataset**  
   2 489 imagens é modesto para treinar ViT; congelar o backbone e treinar só o classificador é adequado.

6. **Recomendações adicionais**  
   • Métricas por classe, com foco em *recall* da minoria.  
   • Avaliar data augmentation específica para *Earthquake* (flip, jitter, rotate).  
   • Considerar `focal loss` ou `weighted CE` para reduzir viés.

Esses insights fundamentam as decisões de pré-processamento, arquitetura (DINO ViT-S/8 congelado) e configuração de treino em CPU. 
