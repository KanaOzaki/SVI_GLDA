# SVI_GLDA

## 実行例
python main.py -d 20ng -f ../data/corpus_20ng.txt -v ../data/vocab_20ng.txt —vec ../data/20ng_vectors_50d.txt -K 50 -t 1024 -k 1.0 -b 16     
/resultにiterationごとのパラメータ(gamma, mu, sigma)が保存される.

## ライブラリ
* scipy
* numpy