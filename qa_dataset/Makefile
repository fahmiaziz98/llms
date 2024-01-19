.PHONY: init download embed training-data

# install Poetry and Python dependencies
init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install
# download news articles and save them in JSON file
download:
	@echo "Downloading data from Alpaca..."
	poetry run python download_news.py