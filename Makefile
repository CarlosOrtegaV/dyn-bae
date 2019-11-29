.PHONY: all clean_raw split_data all_preprocess preprocess_data

FB_URL = "https://www.dropbox.com/s/5zues3doewdwela/fbforum.txt?dl=1"

CALL_URL = "https://www.dropbox.com/s/rika0biieuz4csp/reality_call.txt?dl=1"

# Download data
data/raw/fbforum.txt:
	python dynbae/preprocessing/download.py $(FB_URL) $@
	
data/raw/realitycall.txt:
	python dynbae/preprocessing/download.py $(CALL_URL) $@

# Split facebook-forum data 
split_fb: data/raw/fbforum.txt
	python dynbae/preprocessing/split_fb.py $< data/processed/fb/ --generate_dynamic --training_size=9 --interm_directory=data/intermediate/fb/

split_cal: data/raw/realitycall.txt
	python dynbae/preprocessing/split_cal.py $< data/processed/cal/ --generate_dynamic --training_size=26 --interm_directory=data/intermediate/cal/

## Downsampling
downsample_fb: split_fb
	python dynbae/preprocessing/downsampling.py data/processed/fb/Gsub_dy_test_fb data/processed/fb/ --generate_dynamic

downsample_cal: split_cal
	python dynbae/preprocessing/downsampling.py data/processed/cal/Gsub_dy_test_cal data/processed/cal/ --generate_dynamic

## All
all_preprocess: downsample_fb downsample_cal

all_clean:
	rm -f data/raw/*.txt
	rm -f data/intermediate/fb/*.csv
	rm -f data/intermediate/cal/*.csv
	rm -f data/processed/fb/*.csv
	rm -f data/processed/cal/*.csv

	rm -f data/raw/fb/*.pkl
	rm -f data/raw/cal/*.pkl
	rm -f data/intermediate/fb/*.pkl
	rm -f data/intermediate/cal/*.pkl
	rm -f data/processed/fb/*.pkl
	rm -f data/processed/cal/*.pkl

