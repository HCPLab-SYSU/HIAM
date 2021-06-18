# SHMOD & HZMOD Dataset
In this work, we focus on the metro ridership of 5:30 - 23:30 and utilize the metro OD/DO distribution of the previous four time intervals (15minutes x 4 = 60minutes) to predict the metro OD/DO distribution of future four time intervals (15minutes x 4 = 60minutes). 

Specifically, the data of input sequence consists of incomplete OD matrix(IOD~t~), unfinished order vector(U~t~), and DO matrix(DO~t~) and our online origin-destination prediction could be denoted as follows.

​																![image](https://github.com/GillianZhu/HIAM/blob/master/data/prediction_formula.png)

​													where i=1,2,...,n and j=1,2,...,m. n=4, m=4

We release two datasets named SHMOD and HZMOD, respectively. Each dataset is divided into a training set, a validation set and a testing set, and also contains corresponding metro graph information. For each set, we release four `pkl` files, three for metro OD ridership distribution data, and one for metro DO ridership distribution data.

### 1. Metro OD&DO Matrix

- train.pkl: the OD training set. It is a `dict` that consists of 5 `ndarray`:

```
(1) finished: the incomplete OD matrix of the previous four time intervals. Its shape is [T, n, N, D]. 
(2) unfinished: the unfinished order vector of the previous four time intervals. Its shape is [T, n, N, 1]. 
(3) y: the complete OD matrix of the next four time intervals. Its shape is [T, n, N, D]. 
(4) xtime: the timestamps of x. Its shape is [T, n]. 
(5) ytime: the timestamps of y. Its shape is [T, m].

T = the number of time slices
N = the number of metro stations, i.e, 80/288 (HZMOD/SHMOD)  in our work
n = the length of the input sequence,  i.e, 4 time intervals in our work
m = the length of the output sequence, i.e, 4 time intervals in our work
D = the data dimension of each station, i.e, 26/76 (HZMOD/SHMOD) in our work
```

- train_history_long.pkl: the long-term historical information of the training set. It is a `dict` that consists of 1 `ndarray`:

```
(1) history: the long-term destination distribution matrix of the previous four time intervals. Its shape is [T, n, N, D]. 
```

Note that val_history_long.pkl and test_history_long.pkl are also built based on the long-term historical data on the training set.

- train_history_short.pkl: the short-term historical information of the training set. It is a `dict` that consists of 1 `ndarray`:

```
(1) history: the short-term destination distribution matrix of the previous four time intervals. Its shape is [T, n, N, D]. 
```

- train_do.pkl: the DO training set. It is a `dict` that consists of 4 `ndarray`:

```
(1) do_x: the DO matrix of the previous four time intervals. Its shape is [T, n, N, D]. 
(2) do_y: the DO matrix of the next four time intervals. Its shape is [T, m, N, D]. 
(3) xtime: the timestamps of x. Its shape is [T, n]. 
(4) ytime: the timestamps of y. Its shape is [T, m].
```

Besides, each pkl file has its counterparts on testing set and validation set with similar structure.

### 2. Graph Information

- graph_conn.pkl: the physical topology information of metro system.
