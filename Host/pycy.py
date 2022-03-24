import os
import sys
import mmap
import struct
import numpy as np
from abc import ABC

cma_bytes_reserved = 0


word_fmt = {
    int       : (4, 'I', False),
    
    np.uint8  : (1, 'B', True),
    np.int8   : (1, 'b', True),

    np.uint16 : (2, 'H', True),
    np.int16  : (2, 'h', True),

    np.uint32 : (4, 'I', True),
    np.int32  : (4, 'i', True),
    
    np.uint64 : (8, 'Q', True),
    np.int64  : (8, 'q', True)
}

class DevMem(ABC):
    def __init__(self, base_addr, dtype, length=1, filename = '/dev/mem'):
        if base_addr < 0 or length < 0: 
            raise AssertionError
        self.f = None
        self.dtype = dtype
        self.word, self.fmt, self.is_array = word_fmt[dtype]
        self.mask = ~(self.word - 1)
        self.base_addr = base_addr & ~(mmap.PAGESIZE - 1)
        self.base_addr_offset = base_addr - self.base_addr
        stop = base_addr + length * self.word
        if (stop % self.mask):
            stop = (stop + self.word) & ~(self.word - 1)
        self.length = stop - self.base_addr
        self.fname = filename
        self.f = os.open(self.fname, os.O_RDWR | os.O_SYNC)
        self.mem = mmap.mmap(self.f, self.length, mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=self.base_addr)

    def read(self, offset, length):
        if offset < 0 or length < 0: 
            raise AssertionError
        mem = self.mem
        virt_base_addr = self.base_addr_offset & self.mask
        mem.seek(virt_base_addr + offset)
        if self.is_array:
            data = []
            for i in range(length):
                data.append(struct.unpack(self.fmt, mem.read(self.word))[0])
            return np.asarray(data, dtype=self.dtype)
        else:
            data = struct.unpack(self.fmt, mem.read(self.word))[0]
            return data

    def write(self, offset, din):
        if offset < 0 or (offset & ~self.mask): 
            raise AssertionError
        mem = self.mem
        virt_base_addr = self.base_addr_offset & self.mask
        mem.seek(virt_base_addr + offset)
        if self.is_array:        
            din_array = np.asarray(din, dtype=self.dtype)
            for i in range(len(din_array)):
                mem.write(struct.pack(self.fmt, din_array[i]))
        else:
            mem.write(struct.pack(self.fmt, din))


    def __del__(self):
        if self.f:
            os.close(self.f)



class cma_buffer(DevMem):
    
    def __init__(self, length, dtype):
        global cma_bytes_reserved
        self.num_elements = length
        self.array_index = cma_bytes_reserved
        super(cma_buffer, self).__init__(self.array_index, dtype, self.num_elements)#, filename="/dev/udmabuf0")
        self.udma_phy_addr = 0x38100000
        cma_bytes_reserved += self.num_elements*self.word
        self.write(np.zeros(shape=(self.num_elements,), dtype=dtype))

    def read(self):
        return super().read(0, self.num_elements)

    def write(self, array_in):
        super().write(0, array_in)

    @property
    def physical_address(self):
        return self.array_index#+self.udma_phy_addr

class Device_Driver(DevMem):      
    def __init__(self, base_addr, length):
        super(Device_Driver, self).__init__(base_addr, int, length)

    def read(self, reg_addr):
        return super().read(reg_addr, 1)

    def write(self, reg_addr, value):
        super().write(reg_addr, value)

