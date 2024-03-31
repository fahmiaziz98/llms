### Run ###

run_real_time:
	RUST_BACKTRACE=full poetry run python -m bytewax.run tools.run_stream:build_flow

run_real_time_dev:
	RUST_BACKTRACE=full poetry run python -m bytewax.run "tools.run_stream:build_flow(debug=True)"

run_batch:
	RUST_BACKTRACE=full poetry run python -m bytewax.run -p4 "tools.run_batch:build_flow(latest_n_days=8)"

run_batch_dev:
	RUST_BACKTRACE=full poetry run python -m bytewax.run "tools.run_batch:build_flow(latest_n_days=2, debug=True)"

search:
	poetry run python -m tools.search ${PARAMS}


### Run Docker ###

build:
	@echo "Build docker image"

	docker build -t streaming_pipeline:latest -f deploy/Dockerfile .

run_real_time_docker: 
	@echo "Run docker image"
	
	docker run --rm \
		-e BYTEWAX_PYTHON_FILE_PATH=tools.run_stream:build_flow \
		-e ALPACA_API_KEY=${ALPACA_API_KEY} \
		-e ALPACA_API_SECRET=${ALPACA_API_SECRET} \
		-e QDRANT_API_KEY=${QDRANT_API_KEY} \
		-e QDRANT_URL=${QDRANT_URL} \
		--name streaming_pipeline \
		streaming_pipeline:latest

run_docker_dev:
	@echo "Run docker image"

	docker run -it --rm \
		--env-file .env \
		-e DEBUG=true \
		--name streaming_pipeline \
		streaming_pipeline:latest
