import tensorflow as tf
import numpy as np
import os

def read_my_file_format(filename_queue):
	reader = tf.TextLineReader(skip_header_lines=1)
	key, value = reader.read(filename_queue)

	# Default values, in case of empty columns. Also specifies the type of the
	# decoded result.
	record_defaults = 3*[[""]]+29*[[0.0]]
	#a,b,c,d,LastPrice, Volume, LastVolume, Turnover, LastTurnover,e,f,g,h,AskPrice1,BidPrice1,i,j,k,l,m,n,o,p,AskVolume1,BidVolume1,q,r,s,t,u,v,w,x
	row_list = tf.decode_csv(
    	value, record_defaults=record_defaults)
	LastPrice = row_list[4]
	Volume = row_list[5]
	LastVolume = row_list[6]
	Turnover = row_list[7]
	LastTurnover = row_list[8]
	AskPrice1 = row_list[13]
	BidPrice1 = row_list[14]
	AskVolume1 = row_list[23]#
	BidVolume1 = row_list[24]
	features = tf.stack([Volume, LastVolume, Turnover, LastTurnover,AskPrice1,BidPrice1,AskVolume1,BidVolume1])
	label = LastPrice
	return features, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch


data_list = []
path = "data/m0000"
files = os.listdir(path)
for file in files:
	if not os.path.isdir(file):
		data_list.append(path+"/"+file)

#filename_queue = tf.train.string_input_producer(data_list)
#features, label = read_my_file_format(filename_queue)
features, label = input_pipeline(data_list,2)

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
  example, y = sess.run([features, label])
  print(example,y)
  example, y = sess.run([features, label])
  print(example,y)
  coord.request_stop()
  coord.join(threads)