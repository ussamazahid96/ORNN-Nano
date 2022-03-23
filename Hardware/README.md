## Setup

```bash
sudo apt-get install g++-multilib make
```

## HLS Synthesis of the ORNN accelerator

```
source /opt/intelFPGA_lite/21.1/hls/init_hls.sh
cd HLS
make
cd ..
```

## RBF generation

```
make qsys_edit
```
After you have the plaform designer window, open `Tools > Options > Add...` and add the path of the `HLS/csynth.prj/components/` to the IP path and then synthesize the `.rbf` file by simply running

```
export QUARTUS_ROOTDIR=$(which quartus)/../../
make
```