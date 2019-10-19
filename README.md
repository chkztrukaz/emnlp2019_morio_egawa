# Revealing and Predicting Online Persuasion Strategy with Elementary Units
This is an official implementation for our paper entitled "Revealing and Predicting Online Persuasion Strategy with Elementary Units" at EMNLP 2019.


## ChangeMyView Annotated Dataset
Please download the ChangeMyView annotated dataset (.zip) below:
- http://katfuji.lab.tuat.ac.jp/nlp_datasets/

Unzip and place the downloaded dataset in the top directory.


## Run

##### Donwload GloVe
```
./setup_dataset.sh
```

##### Run Train
```
python blc_trainer.py --gpu {GPU No. or -1 if using CPU}
```

Note that evaluation results change for each run because we use random train/test split and initializations per run.


## Citation

If you use our dataset or code, please cite following papers.

#### 1. [EMNLP 2019] Revealing and Predicting Online Persuasion Strategy with Elementary Units

This paper is a pilot study which analyzes EUs and shows the method to predict EUs by neural networks.

```
@inproceedings{morio-etal-2019,
    title = "Revealing and Predicting Online Persuasion Strategy with Elementary Units",
    author = "Morio, Gaku  and
      Egawa, Ryo  and
      Fujita, Katsuhide",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and 9th International Joint Conference on Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    pages = "to appear",
}
```

#### 2. [ACL-SRW 2019] Annotating and Analyzing Semantic Role of Elementary Units and Relations in Online Persuasive Arguments

This paper describes detailed annotation process of arguments (EUs and Support/Attack links) in ChangeMyView and conducts some  examinations.

```
@inproceedings{egawa-etal-2019,
    title = "Annotating and Analyzing Semantic Role of Elementary Units and Relations in Online Persuasive Arguments",
    author = "Egawa, Ryo  and
      Morio, Gaku  and
      Fujita, Katsuhide",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-2059",
    pages = "422--428",
}
```
