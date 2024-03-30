export_requirements:
	@echo "Exporting requirements..."

	if [ -f requirements.txt ]; then rm requirements.txt; fi
	poetry export -f requirements.txt --output requirements.txt --without-hashes


upload_dataset:
	beam volume upload qa_dataset dataset

train_beam: 
	@echo "Running training pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/train.py:train -d '{"config_file": "configs/training_config_sft.yaml", "output_dir": "./output", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'


infer_beam: 
	@echo "Running inference pipeline on Beam using the production config..."

	BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_run.py:infer -d '{"config_file": "configs/inference_config.yaml", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'
