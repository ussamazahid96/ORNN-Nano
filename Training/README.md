# Training ORNN

## Install GeoTorch

```bash
pip install git+https://github.com/Lezcano/geotorch/
```

## Run the Training on AudioMNIST

Download the dataset from https://github.com/soerenab/AudioMNIST/tree/master/data and place it under `dataset/`. Make sure that the structure of the folders is like `dataset/01/`, `dataset/02` etc. 

```bash
python main.py --ib 8 --wb 4 --rb 4 --ab 8
```

## Export the Model

```bash
python main.py --ib 8 --wb 4 --rb 4 --ab 8 --resume <path to model best.tar> --export
```


