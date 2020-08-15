# Quin
An easy to use framework for scalable semantic search.

[<a href="https://quin.algoprog.com">Demo</a>] - [<a href="https://www.researchgate.net/publication/342815574_Latent_Retrieval_for_Large-Scale_Fact-Checking_and_Question_Answering_with_NLI_training">Paper</a>] - [<a href="https://towardsdatascience.com/building-a-semantic-search-engine-for-large-scale-fact-checking-and-question-answering-9aa356632432">Blog post</a>]

## Usage

1) Download the model weights (encoder, passage ranker, NLI) from <a href="https://drive.google.com/file/d/1dBMCxa7xYvGNMZGyonOQO1nyoB_CgXAe/view?usp=sharing">here</a> and extract them into the models/weights folder.

2) Initialise Quin with an index path:
```python3
q = Quin(index_path='index')
```

3) Index a list of documents:
```python3
q.index_documents(documents=[
    'Document text 1',
    'Document text 2'
])
```

4) Serve a Flask API:
```python3
q.serve()
```
