run: preprocess train validate test

preprocess:
	python3.6 ../preprocess_masked.py -maxpatch 1000000
	python3.6 ../preprocess_easy.py
	python3.6 ../preprocess_multi.py
	python3.6 ../preprocess_more.py

train:
	python3 ../train.py

validate:
	python3 ../validate.py

test:
	python3 ../test.py

clean:
	rm -rf preprocess_masked
	rm -rf preprocess_easy
	rm -rf preprocess_multi
	rm -rf preprocess_more
	rm -rf train_output
	rm -rf validate_output
	rm -rf test_output
