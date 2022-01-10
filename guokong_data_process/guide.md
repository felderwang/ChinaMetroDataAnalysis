### 使用手册

1. convert_excel2h5.py

* crearte_year_h5()

```
将原始excel文件，转化成h5格式。列为datetime和规定的site_list。不做数据变换。
```


2. guokong_data_process.py

* see_df()

```
只是看一下h5。
```


* deal_guokong()

```
将原始数据处理进行变换。
```


```
该版本提供的'log_exclude3std_reshape2n01'，表示的是首先x=log(x+1)，进行对数变换，然后移除变换后的3std外的数据。最后按照移除3std后的最大值最小值，将数据线性变换到[0,1]区间。（min=min(0, min(data))
```


* divide_hdf2xlsx4rstlplus()

```
将h5文件划分为分立的xlsx文件，供r语言stlplus处理。分立是指，各个key各个站点都分开。
```


* combine_singleyearh5_2_multiyearh5()

```
将每年的h5文件合并成多年的h5，这样可以使得数据变换时使用统一的参数，同时stlplus分解也需要更长的序列。
```
