# Training ORNN

## Install GeoTorch
```bash
pip install git+https://github.com/Lezcano/geotorch/
```

## Run the Training

```bash
python main.py --model QORNN --ib 4 --wb 4 --rb 4 --ab 8
```


## Export the Model

```bash
python main.py --model QORNN --ib 4 --wb 4 --rb 4 --ab 8 --resume <path to model best.tar> --export
```

