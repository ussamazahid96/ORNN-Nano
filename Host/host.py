import numpy as np
from pycy import *
import time

SEQ_LEN = 35
ELEMENTS_IN = 16*SEQ_LEN*8//64 # 64
ELEMENTS_OUT = 16*SEQ_LEN*16//64 # 1024
input_buffer = 0x20
output_buffer = 0x28
seq_len = 0x30
start = 0x8
status = 0x0

def main():

    inp = np.loadtxt("../Training/export/QORNN_W4R4/test_image.txt")
    inp = np.clip(inp, -1, 1-2**-7)
    inp = np.round(inp*2**7)
    inp = inp.astype(np.int8).view(np.uint64)

    array_a = cma_buffer(ELEMENTS_IN, np.uint64)
    array_b = cma_buffer(ELEMENTS_OUT, np.uint64)
    ACCL_TOP_csr = Device_Driver(0xff200000, 64)
    ACCL_TOP_csr.write(input_buffer,  array_a.physical_address)
    ACCL_TOP_csr.write(output_buffer, array_b.physical_address)
    ACCL_TOP_csr.write(seq_len, SEQ_LEN)
    ACCL_TOP_csr.write(0x10, 0)
    ACCL_TOP_csr.write(0x18, 0x00)

    array_a.write(inp)
    ACCL_TOP_csr.write(start, 1)
    start_time = time.time()
    while(not (ACCL_TOP_csr.read(0x18) & 0x2)):
        pass
    pred = array_b.read().view(np.int16)
    pred = pred/2**10
    max_val = 0
    for i in range(SEQ_LEN):
        max_val = 0
        max_score = pred[i*16]
        for j in range(16):
            if j<10:
                if pred[i*16+j] >= max_score:
                    max_score = pred[i*16+j]
                    max_val = j
    print("Pred = {}".format(max_val))

if __name__=='__main__':
    main()



