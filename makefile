# Makefile for automatic execution of data_preprocess.py and model_train.py

# Load configuration
# Define targets
.PHONY: preprocess train

# Target to preprocess the data
preprocess:
	@echo "Running data preprocessing..."
	@python data_preprocess.py

# Target to train the model
train:
	@echo "Training the model..."
	@python model_train.py
