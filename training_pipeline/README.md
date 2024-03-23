# Training Pipeline
Training pipeline that:

- loads a proprietary Q&A dataset
- fine-tunes an open-source LLM using QLoRA
- logs the training experiments on Comet ML's experiment tracker & the inference results on Comet ML's LLMOps dashboard
- stores the best model on Comet ML's model registry

The training pipeline is deployed using [Beam](https://docs.beam.cloud/getting-started/installation) as a serverless GPU infrastructure.

## Usage for Development
Before you start, you must have an API Key from [Comet-ML](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/getting-started/) and install the [Beam SDK](https://docs.beam.cloud/getting-started/installation). Additionally, you need to set up the API key by configuring it in the appropriate environment variable.

Before you begin using the training pipeline, please ensure you have access to the Beam cloud platform. Once logged in to your Beam account, proceed to create a volume:

1. Go to the Volumes section.
2. Click "New Volume" in the top right corner.
3. Choose "qa_dataset" for Volume Name and "Shared" for Volume Type.

----

- Prepare credentials:

    ```bash
    cp .env.example .env
    ```

- Replace your api, workspace, project name
    ```bash
    COMET_API_KEY = "comet-api-key"
    COMET_WORKSPACE = "your-workspace"
    COMET_PROJECT_NAME = "your-project-name"
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