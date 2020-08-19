# Quin
An easy to use framework for scalable semantic search.

[<a href="https://quin.algoprog.com">Demo</a>] - [<a href="https://www.researchgate.net/publication/342815574_Latent_Retrieval_for_Large-Scale_Fact-Checking_and_Question_Answering_with_NLI_training">Paper</a>] - [<a href="https://towardsdatascience.com/building-a-semantic-search-engine-for-large-scale-fact-checking-and-question-answering-9aa356632432">Blog post</a>] - [<a href="https://docs.google.com/presentation/d/1QpDF4xWgLSF-2DC1q5M_9MN7pASn-2T6NgKkhJ-NTZ8/edit?usp=sharing">Presentation</a>] - [<a href="https://archive.org/details/factual-nli">Factual-NLI Dataset</a>]

<img src="https://miro.medium.com/max/1400/1*-LaR_PfEbfJcH_BpD0Sptg.png" width="500"/>

## Usage

1) Download the model weights (encoder, passage ranker, NLI) from <a href="https://drive.google.com/file/d/1dBMCxa7xYvGNMZGyonOQO1nyoB_CgXAe/view?usp=sharing">here</a> and extract them into the models/weights folder.

2) Install the required packages:
```
pip3 install -r requirements.txt
```

3) Initialise Quin with an index path (Quin class from quin.py):
```python3
q = Quin(index_path='index')
```

4) Index a list of documents:
```python3
q.index_documents(documents=[
    'Document text 1',
    'Document text 2'
])
```

5) Serve a Flask API:
```python3
q.serve()
```
