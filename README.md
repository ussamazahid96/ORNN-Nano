# ORNN-Nano
Orthognal Recurrent Neural Network on DE10-Nano.

**This project is submitted to the Intel [InnovateFPGA contest 2021-2022](https://www.innovatefpga.com/portal/)**.

This model is trained on audioMNIST dataset taken from [here](https://github.com/soerenab/AudioMNIST) i.e., given a speech sample of a digit it predicts the number 0-9. 

Similarly, in the `main` branch, the model is trained on MNIST dataset, it predicts the digit 0-9 given an image file.


## Running demo on DE10-Nano

Install the [FPGA-SoC-Linux](https://github.com/ikwzm/FPGA-SoC-Linux) by following the given instructions.

Make sure that the shared internet connection profile is setup from your host machine to the connected DE10-Nano. From you host machine run:

```
echo <your ip mask e.g 10.42.0. or 192.168.1.>{1..254} | xargs -P255 -n1 ping -s1 -c1 -W1 | grep ttl # To find the IP of the connected devices to the machine
ssh -L 9090:localhost:9090 fpga@<ip of the board> # you have to try all the listed IP to find the correct one :(
password is "fpga"
```

### Install dependencies on DE10-Nano

```
sudo apt update
sudo apt install -y rsync
sudo apt install libffi-dev
sudo apt-get install libjpeg-dev zlib1g-dev
pip3 install notebook pillow pybind11 cython python_speech_features
```

### Run the inference Using python interpreter

Clone the project to the board directly using:

```
git clone https://github.com/ussamazahid96/ORNN-Nano.git
```

Or to copy the whole ORNN-Nano project to DE10 using rsync from your host machine run the follow from you host machine:

```
rsync -avP <Path to the ORNN-Nano folder>/ORNN-Nano fpga@<ip of the DE10>~/
```

Next upload the `.rbf` file and run the `host.py` to lauch the inference.

```
sudo -s
make install
python3 host.py
make uninstall # to unload the .rbf
```

### Running the Notebook

Launch the notebook from the `Host` directory and open the given url in the browser of the host machine and run the `ORNN-Demo.ipynb`.

```
jupyter notebook --allow-root --no-browser --port=9090
```

## Acknowledgements

Inspired from:

https://github.com/Xilinx/LSTM-PYNQ


https://github.com/Xilinx/finn-hlslib





