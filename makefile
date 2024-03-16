# Makefile for automatic execution of data_preprocess.py and model_train.py

# Load configuration
CONFIG := $(shell cat config.yml)
FOLDER_WATCH_PATH := $(shell echo $(CONFIG) | jq -r '.folder_watch.path')

# Define targets
.PHONY: watch preprocess train

# Target to watch for changes in the new_data folder
watch:
	@echo "Watching for changes in $(FOLDER_WATCH_PATH)"
	@fswatch -o $(FOLDER_WATCH_PATH) | xargs -n1 -I{} make preprocess train

# Target to preprocess the data
preprocess:
	@echo "Running data preprocessing..."
	@python data_preprocess.py

# Target to train the model
train:
	@echo "Training the model..."
	@python model_train.py
