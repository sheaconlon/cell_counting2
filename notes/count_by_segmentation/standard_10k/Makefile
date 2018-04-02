run: preprocess train validate

preprocess:
	python3 ../preprocess_masks_and_counts.py -maxpatch 10000
	python3 ../preprocess_counts_easy.py
	python3 ../preprocess_counts_multicondition.py
	python3 ../preprocess_multicondition_pieces.py
	python3 ../preprocess_more.py

train:
	python3 ../train.py -metricexamples 1000

validate:
	python3 ../validate.py

clean:
	rm -rf preprocess_masks_and_counts_output
	rm -rf preprocess_counts_easy_output
	rm -rf preprocess_counts_multicondition_output
	rm -rf preprocess_multicondition_pieces_output
	rm -rf preprocess_more_output
	rm -rf train_output
