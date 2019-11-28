.PHONY: all clean_raw split_data all_preprocess preprocess_data

TRAIN_URL = "https://www.dropbox.com/s/a20trb1h9h29n7p/insurance_fraud_train.csv?dl=1"

TEST_URL = "https://www.dropbox.com/s/9onwy46nccyf1i6/insurance_fraud_test.csv?dl=1"

CHURN1_URL = "https://www.dropbox.com/s/65q5v8oyoxl52bu/CHILE_POSTPAID_PROCESSED.csv?dl=1"

data/raw/train_insur.csv:
	python labcleaner/preprocessing/download.py $(TRAIN_URL) $@
	
data/raw/test_insur.csv:
	python labcleaner/preprocessing/download.py $(TEST_URL) $@

# Download Raw Data
data/raw/churn1.csv:
	python labcleaner/preprocessing/download.py $(CHURN1_URL) $@

# Generate Noise Label
data/intermediate/noisy_churn1.csv: data/raw/churn1.csv
	python labcleaner/preprocessing/noise_generator.py --noise_level=0.25 --excel=data/intermediate/noisy_churn1.xlsx $< $@ 

# Split original data 
split_data: data/raw/churn1.csv
	python labcleaner/preprocessing/split_churn.py $< data/intermediate/ --excel

# Split noisy data 
split_noisy_data: data/intermediate/noisy_churn1.csv
	python labcleaner/preprocessing/split_churn.py $< data/intermediate/ --excel

## Preprocessing of original data
preprocess_train_data: split_data
	python labcleaner/preprocessing/preprop_churn.py data/intermediate/train.pkl data/processed/trans_train.pkl --excel=data/processed/trans_train.xlsx
	
preprocess_test_data: split_data
	python labcleaner/preprocessing/preprop_churn.py data/intermediate/test.pkl data/processed/trans_test.pkl --excel=data/processed/trans_test.xlsx

preprocess_data: preprocess_train_data preprocess_test_data

# Preprocessing of noisy data
preprocess_train_noisy_data: split_noisy_data
	python labcleaner/preprocessing/preprop_churn.py data/intermediate/train.pkl data/processed/trans_noisy_train.pkl --excel=data/processed/noisy_trans_train.xlsx
	
preprocess_test_noisy_data: split_noisy_data
	python labcleaner/preprocessing/preprop_churn.py data/intermediate/test.pkl data/processed/trans_noisy_test.pkl --excel=data/processed/noisy_trans_test.xlsx

preprocess_noisy_data: preprocess_train_noisy_data preprocess_test_noisy_data

## All
all_preprocess: preprocess_data preprocess_noisy_data

all_clean:
	rm -f data/raw/*.csv
	rm -f data/intermediate/*.pkl
	rm -f data/processed/*.pkl
	rm -f data/intermediate/*.xlsx
	rm -f data/processed/*.xlsx

all_churn: data/raw/churn1.csv split_data
