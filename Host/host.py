import numpy as np
from pycy import *
import time

SEQ_LEN = 32
ELEMENTS_IN = 32*SEQ_LEN*4//64 # 64
ELEMENTS_OUT = 16*SEQ_LEN*16//64 # 1024
input_buffer = 0x20
output_buffer = 0x28
seq_len = 0x30
start = 0x8
status = 0x0

def load_mnist_labels():
   labels = []
   with open("../Hardware/HLS/utils/mnist_data/t10k-labels.idx1-ubyte","rb") as lbl_file:
       #read magic number and number of labels (MSB first) -> MNIST header
       magicNum = int.from_bytes(lbl_file.read(4), byteorder="big")
       countLbl = int.from_bytes(lbl_file.read(4), byteorder="big")
       #now the labels are following byte-wise
       for idx in range(countLbl):
           labels.append(int.from_bytes(lbl_file.read(1), byteorder="big"))
       lbl_file.close()
   return labels

def load_mnist_images():
   with open("../Hardware/HLS/utils/mnist_data/t10k-images.idx3-ubyte","rb") as imgs:
      data = np.frombuffer(imgs.read(), np.uint8)

   header, data = data[:16], data[16:]
   data = data.reshape(-1, 28, 28)/255.
   data = 2*data-1
   data = np.pad(data, ((0,0),(2,2),(2,2)), constant_values=(-1, -1))
   return data


def preprocess_image(input_image):
   input_image = np.clip(input_image, -1, 1-2**-3)
   input_image = np.round(input_image*2**3)
   input_image = input_image.reshape(-1)
   q_img = np.zeros(shape=(512,), dtype=np.uint8)
   index = 0
   for i in range(0, len(input_image), 2):
      elem1 = int(input_image[i]) & 0xf
      elem2 = int(input_image[i+1]) & 0xf
      val = (elem2 << 4) | (elem1)
      q_img[index] = val
      index+=1
   q_img = q_img.view(np.uint64)
   return q_img

def main():
   mnist_images = load_mnist_images()
   mnist_labels = load_mnist_labels()

   array_a = cma_buffer(ELEMENTS_IN, np.uint64)
   array_b = cma_buffer(ELEMENTS_OUT, np.uint64)
   ACCL_TOP_csr = Device_Driver(0xff200000, 64)
   ACCL_TOP_csr.write(input_buffer,  array_a.physical_address)
   ACCL_TOP_csr.write(output_buffer, array_b.physical_address)
   ACCL_TOP_csr.write(seq_len, SEQ_LEN)
   ACCL_TOP_csr.write(0x10, 0)
   ACCL_TOP_csr.write(0x18, 0x00)

   correct = 0
   time_tot = 0
   for k in range(10000):
      img = mnist_images[k]
      lab = mnist_labels[k]
      q_img = preprocess_image(img)
      array_a.write(q_img)
      ACCL_TOP_csr.write(start, 1)
      start_time = time.time()
      while(not (ACCL_TOP_csr.read(0x18) & 0x2)):
         pass
      time_end = time.time()-start_time
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
      if lab == max_val:
         correct += 1
      time_tot += time_end
      print("Accuracy on [{}/10000] images = {:.2f}%".format(k+1, (correct/10000)*100), end='\r')
   print("\nImages per second (FPS) = {:.2f}".format(10000/time_tot))

if __name__=='__main__':
   main()
    
    
    
