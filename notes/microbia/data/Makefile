all: clean download preprocess

clean:
	rm -rf raw.zip
	rm -rf raw
	rm train_images.npy
	rm train_labels.npy
	rm test_images.npy
	rm test_labels.npy

download:
	wget -O raw.zip https://s3.amazonaws.com/ludingtonlab/microbia.zip
	unzip raw.zip
	
preprocess:	
	python3 create.py
