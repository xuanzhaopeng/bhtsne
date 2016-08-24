import caffe
from caffe.proto import caffe_pb2
import lmdb
import leveldb
import numpy as Math
import pylab as Plot

def readLmdb(lmdb_file, model_def_path, model_weights_path, layer_name, length):
        caffe.set_mode_gpu()
        net = caffe.Net(model_def_path, model_weights_path, caffe.TEST)
        lmdb_env = lmdb.open(lmdb_file)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe_pb2.Datum()
        labels = Math.array([])
	counter = 1
	X = Math.ndarray((0,length)) 
	for key, value in lmdb_cursor:
    		datum.ParseFromString(value)
                labels = Math.append(labels, datum.label)
  		net.blobs['data'].data[...] = caffe.io.datum_to_array(datum)
		net.forward()
		feature_arr = Math.array(net.params[layer_name][1].data.copy())
		X = Math.append(X, [feature_arr], axis=0)
 		if counter % 1000 == 0:
			print("Read already %d images" %(counter))
		counter = counter + 1
        print (X.shape)
	print (labels.shape)
	Math.savetxt('in.out', X, delimiter="\t")
	
if  __name__ == "__main__":
        readLmdb('/home/ubuntu/caffe/examples/mnist/mnist_test_lmdb', 
		'/home/ubuntu/caffe/robin/deploy.prototxt', 
		'/home/ubuntu/caffe/robin/output/ae_iter_20000.caffemodel', 
		'ip2encode',
                30)
