# Quin
![GitHub](https://img.shields.io/github/license/algoprog/Quin)

An easy to use framework for large-scale fact-checking and question answering.

[<a href="https://quin.algoprog.com">Demo</a>] - [<a href="https://archive.org/details/factual-nli">Factual-NLI Dataset</a>]

## Usage

1) Download the model weights and extract them into the models/weights folder:

 - <a href="https://drive.google.com/file/d/15Txw44izeEHCzzXIpxwVXFvNz_-_kng-/view?usp=sharing">NLI model</a> and
 - <a href="https://drive.google.com/file/d/1qsDPreap_26mL3UFDEyVPoe9ygbniLx9/view?usp=sharing">Dense Encoder M</a> (multitask for QA and Fact-Checking) or 
 - <a href="https://drive.google.com/file/d/1G3eMkVrd-lA5cbWhwme8f5RpplacTMvF/view?usp=sharing">Dense Encoder FC</a> (single-task for Fact-Checking) or 
 - <a href="https://drive.google.com/file/d/1uco7t8drHuagiS6hwNFQlayAhYVIFyfY/view?usp=sharing">Dense Encoder QA</a> (single-task for Question Answering)

2) Install the required packages:
```
pip3 install -r requirements.txt
```

3) Index a list of documents:
```
python quin.py --index example_docs.jsonl
```

4) Serve a Flask API:
```
python quin.py --port 1234
```

## References

```
@inproceedings{samarinas2021improving,
  title={Improving Evidence Retrieval for Automated Explainable Fact-Checking},
  author={Samarinas, Chris and Hsu, Wynne and Lee, Mong Li},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Demonstrations},
  pages={84--91},
  year={2021}
}

@inproceedings{samarinas2020latent,
  title={Latent Retrieval for Large-Scale Fact-Checking and Question Answering with NLI training},
  author={Samarinas, Chris and Hsu, Wynne and Lee, Mong Li},
  booktitle={2020 IEEE 32nd International Conference on Tools with Artificial Intelligence (ICTAI)},
  pages={941--948},
  year={2020},
  organization={IEEE}
}
```

## License

Quin is licensed under <a href="https://github.com/algoprog/Quin/blob/master/LICENSE">MIT License</a>, and the Factual-NLI dataset under <a href="https://creativecommons.org/licenses/by/4.0/">Attribution 4.0 International (CC BY 4.0) license</a>.
