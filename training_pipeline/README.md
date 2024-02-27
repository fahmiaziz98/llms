
## Usage for Development
Before you start, you must have an API Key from [Comet-ML](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/getting-started/) and install the [Beam SDK](https://docs.beam.cloud/getting-started/installation). Additionally, you need to set up the API key by configuring it in the appropriate environment variable.

- Prepare credentials:

    ```bash
    cp .env.example .env
    ```

- This command will build the dataset and push it to the Beam cloud. Run this command to prepare the dataset before training the model:

    ```bash
    make upload_dataset
    ```

- Train the model on the Beam cloud using this command. It will initiate the model training process using the prepared dataset:

    ```bash
    make train_beam
    ```

- Once the model is trained, you can use this command to perform inference with the trained model:

    ```bash
    make infer_beam
    ```