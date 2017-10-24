all: clean preprocess

install:
	sudo -H pip install openpyxl

clean:
	rm -rf raw
	rm -rf train
	rm -rf test

download:
	wget -O raw.zip https://s3.amazonaws.com/ludingtonlab/easy.zip
	unzip raw.zip

preprocess:
	mkdir train
	mkdir train/images
	mkdir test
	mkdir test/images
	python create.py