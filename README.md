# UCR time series dataset loader

This is a simple loader for the [UCR time series dataset](https://www.cs.ucr.edu/~eamonn/time_series_data/).

## Install
```
pip install git+https://github.com/walwe/ucrloader.git#egg=ucrloader
```


## Usage

```
loader = UCRLoader(ucr_data_dir)
for name in loader.names:
    data = loader.load(name)
    # Access data
    print(data.train_labels)
    print(data.train_data)
    print(data.test_labels)
    print(data.test_data)
```


## Adjusted labels

Some of the datasets contain labels not in the range `[0 - X]`.
The UCRData object therefore contains adjusted labels e.g. as required by pytorch.
```
data.test_labels_adjusted
data.train_labels_adjusted
```

The original label can be obtained by accessing `UCRData.unique_labels`: 
```
data.unique_labels[data.test_labels_adjusted[0]]
```
