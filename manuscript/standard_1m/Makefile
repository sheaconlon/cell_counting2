run: preprocess train validate test

preprocess:
	python3.6 ../preprocess_masked.py -maxpatches 1000000
	python3.6 ../preprocess_easy.py
	python3.6 ../preprocess_multi.py
	python3.6 ../preprocess_more.py
	python3.6 ../preprocess_pinned.py

train:
	python3.6 ../train.py
	python3.6 ../train_svm.py

validate:
	python3.6 ../validate.py

test:
	python3.6 ../test.py
	python3.6 ../test_pinned.py

clean:
	rm -rf preprocess_masked
	rm -rf preprocess_easy
	rm -rf preprocess_multi
	rm -rf preprocess_more
	rm -rf preprocess_masked
	rm -rf preprocess_pinned
	rm -rf train
	rm -rf train_svm
	rm -rf validate
	rm -rf test
	rm -rf test_pinned
