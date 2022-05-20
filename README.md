# Information Retrieval Programming Assignment - BM25 Search Engine

## Personal Information

| Name                 | Wang Yuyang |
| -------------------- | ----------- |
| UCD Student Number:  | 19206226    |
| BJUT Student Number: | 19372316    |

## Features
* Implemented BM25 model and Precision, Recall, Precision@10, R-Precision, MAP, b_pref, NDCG for evaluation.
* Automatically evaluate the program and print the evaluation metrics.
* Automatically build and dump the index to a file, no need to read the whole dataset a second time.
* Allowing compression to reduce the index file(from 211 MB -> 155 MB for large corpus, but this will affect the speed of saving and loading, so by default this feature is not enabled.)
* Pre-calculated BM25 score, thus achieves a higher query performance.


## Instructions
I have attempted both of the small and large corpus, so there is a separated program for each of them.
The program name for each of the corpus are the same -- `search.py`

### Small corpus
For small corpus, please switch to the director using this commends:
```shell
cd small-corpus/
```
and run the program:
#### Automatic evaluation
```shell
python search.py -m evaluation
```
the output will be like:
```shell
> python search.py -m evaluation     
Loading index from file...
Build or loading index cost: 0.05389000000000001
Written to file: output.txt
Query cost: 0.20161800000000002
Precision    (0.41186858796527687)
Recall       (0.44108936779729624)
Precision@10 (0.26044444444444453)
R-Precision  (0.35333145124990645)
MAP          (0.33289896356986537)
b_pref       (0.3315311255637636)
NDCG_score   (0.3703139179229204)
Evaluate cost: 0.008191000000000004
```
#### Input query manually
input queries manually:
```shell
python search.py -m manual
```
the output will be like:
```shell
> python search.py -m manual
Loading index from file...
Build or loading index cost: 0.047557999999999996

Please input query.
>>> 
```
once you finished input, you can press enter and the program should display the result immediately:
```shell
Please input query.
>>> what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft
Time <Query>: 0.0016260000000000024
Query 1: what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft
  Query ID         Q0                         Doc ID    Ranking                Score           Student ID
         1         Q0                             51          1            27.834510             19206226
         1         Q0                            486          2            26.818613             19206226
         1         Q0                             12          3            23.858651             19206226
         1         Q0                            184          4            22.988724             19206226
         1         Q0                            878          5            21.514560             19206226
         1         Q0                            665          6            18.871190             19206226
         1         Q0                            573          7            18.665324             19206226
         1         Q0                            944          8            17.101055             19206226
         1         Q0                            141          9            16.551875             19206226
         1         Q0                            746         10            16.203965             19206226
         1         Q0                             78         11            16.123421             19206226
         1         Q0                             14         12            15.558147             19206226
         1         Q0                            453         13            14.795982             19206226
         1         Q0                            329         14            14.478334             19206226
         1         Q0                             13         15            14.468891             19206226

```




### Large corpus

For large corpus, please switch to the director using this commends:
```shell
cd large-corpus/
```
and run the program:
#### Automatic evaluation
```shell
python search.py -m evaluation
```
the output will be like:
```shell
> python search.py -m evaluation
Loading index from file...
Build or loading index cost: 3.209099
Written to file: output.txt
Query cost: 0.8714900000000001
Precision    (0.4235177378905466)
Recall       (0.896058451391523)
Precision@10 (0.5827160493827157)
R-Precision  (0.5296578389467479)
MAP          (0.5551753930814257)
b_pref       (0.40483167942246195)
NDCG_score   (0.5564884484005923)
Evaluate cost: 0.030048999999999992
```
#### Input query manually
input queries manually:
```shell
python search.py -m manual
```
the output will be like:
```shell
> python search.py -m manual
Loading index from file...
Build or loading index cost: 3.28638

Please input query.
>>> 
```
once you finished input, you can press enter and the program should display the result immediately:
```shell
Please input query.
>>> describe history oil industry
Time <Query>: 0.00796000000000019
Query 1: describe history oil industry
  Query ID         Q0                         Doc ID    Ranking                Score           Student ID
         1         Q0               GX232-43-0102505          1             9.044560             19206226
         1         Q0              GX255-56-12408598          2             8.675164             19206226
         1         Q0               GX229-87-1373283          3             8.621311             19206226
         1         Q0               GX253-41-3663663          4             8.543551             19206226
         1         Q0               GX064-43-9736582          5             8.511779             19206226
         1         Q0              GX268-35-11839875          6             8.482392             19206226
         1         Q0              GX231-53-10990040          7             8.411094             19206226
         1         Q0              GX262-28-10252024          8             8.351232             19206226
         1         Q0               GX063-18-3591274          9             8.346404             19206226
         1         Q0              GX263-63-13628209         10             8.226278             19206226
         1         Q0               GX253-57-7230055         11             8.225637             19206226
         1         Q0              GX262-86-10646381         12             8.145205             19206226
         1         Q0              GX000-48-10208090         13             8.140996             19206226
         1         Q0              GX128-96-12152039         14             8.137248             19206226
         1         Q0              GX255-59-12399984         15             8.127072             19206226

```
