import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, bufSize=10):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
            "exists and cannot be overwritten. Manually delete "
            "the file before continuing.", outputPath)
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset('images', dims, dtype="float")
        self.masks = self.db.create_dataset("masks", dims, dtype="float")
        self.bufSize = bufSize
        self.buffer = {"data": [], "mask": []}
        self.idx = 0

    def add(self, rows, masks):
        self.buffer["data"].extend(rows)
        self.buffer["masks"].extend(masks)
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["masks"]
        self.idx = i
        self.buffer = {"data": [], "masks": []}

    def storeClassLabels(self, classLabels):
        #dt = h5py.special_dtype(vlen=unicode)
        labelSet = self.db.create_dataset("label_names",(len(classLabels), ), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()

