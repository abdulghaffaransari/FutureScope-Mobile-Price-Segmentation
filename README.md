# Predicting Mobile Phone Price Range using Machine Learning on Amazon SageMaker

## Introduction

This project demonstrates how to build, train, and deploy a machine learning model on **Amazon SageMaker** to predict the price range of mobile phones based on their specifications. It provides a complete end-to-end machine learning pipeline, from preprocessing data to deploying a fully functional RESTful endpoint for inference.

The model uses a **Random Forest Classifier** to classify mobile phones into four price range categories (0: Low cost, 1: Medium cost, 2: High cost, 3: Very high cost) based on features such as battery power, RAM, screen resolution, and others.

---

## Problem Statement

The goal of this project is to predict the **price range** of mobile phones using their technical specifications. This classification task helps in understanding how different features impact the cost of mobile phones. For example, high RAM and a better screen resolution may contribute to a higher price range.

The dataset includes features such as:
- **battery_power**: Total energy capacity of the battery (mAh).
- **clock_speed**: Processor speed (GHz).
- **px_height, px_width**: Screen resolution (pixels).
- **ram**: Total RAM (MB).
- **n_cores**: Number of CPU cores.
- **price_range**: The target variable with values {0, 1, 2, 3}.

---

## Dataset Overview

The dataset used is structured and labeled, making it ideal for a supervised machine learning task. It includes the following:
- **Features**: Various specifications of mobile phones (e.g., battery power, processor speed, RAM, etc.).
- **Target**: The price range, which is categorized into four classes.

---

## Project Workflow

### 1. **Data Preparation**
- **Dataset Splitting**: The dataset is split into training (85%) and testing (15%) datasets.
- **Preprocessing**: The data is preprocessed to ensure compatibility with the training process. The resulting datasets are uploaded to **Amazon S3** for further processing.

### 2. **Model Training**
- **Random Forest Classifier**: A robust and interpretable machine learning model is trained on the dataset using SageMaker's `scikit-learn` container.
- **Hyperparameters**: Configurable hyperparameters like `n_estimators` and `random_state` are used to fine-tune the model's performance.
- **Monitoring**: The SageMaker console is used to monitor the training process in real time.

### 3. **Model Deployment**
- The trained model is deployed as a RESTful endpoint on SageMaker.
- **Endpoint**: The endpoint allows real-time predictions by sending requests with phone specifications.

### 4. **Model Evaluation**
- The model's performance is evaluated using metrics such as **accuracy** and a detailed **classification report**.
- Example predictions are made to verify the endpoint's functionality.

### 5. **Resource Cleanup**
- The endpoint is deleted after testing to avoid unnecessary costs.
- Model artifacts and datasets are stored securely in Amazon S3.

---

## Tools and Technologies Used

- **Amazon SageMaker**: For training, deploying, and managing the machine learning model.
- **Python**: For data preprocessing, training, and inference scripts.
- **scikit-learn**: Machine learning library used for training and evaluation.
- **boto3**: AWS SDK for Python for interacting with Amazon S3 and SageMaker.
- **Amazon S3**: For storing datasets and trained model artifacts.

---

## Key Features

1. **End-to-End Machine Learning Workflow**:
   - From data preparation to deployment and testing.
   - Covers training, evaluation, and real-time inference.

2. **Cost Optimization**:
   - Utilizes **spot instances** in SageMaker for cost-efficient training.
   - Includes resource cleanup to avoid unnecessary costs.

3. **Custom Inference Script**:
   - `script.py` is designed to preprocess input data, train the model, and perform predictions.

4. **Dynamic Role and Resource Management**:
   - Retrieves SageMaker roles dynamically for seamless integration.
   - Uses modular and reusable code for session management and deployment.

---

## How to Run the Project

### Prerequisites
1. An **AWS account** with appropriate permissions for SageMaker, S3, and IAM.
2. A SageMaker Notebook instance with the necessary Python libraries installed.

### Step-by-Step Guide
1. **Clone the Repository**:
   - Clone the project repository to your SageMaker Notebook instance.

2. **Upload the Dataset**:
   - Upload the dataset to the SageMaker notebook and preprocess it.

3. **Data Preparation**:
   - Use `train_test_split` to split the dataset into training and testing sets.
   - Save the split datasets (`train-V-1.csv` and `test-V-1.csv`) locally and upload them to S3.

4. **Train the Model**:
   - Configure the `SKLearn` estimator and specify the entry point (`script.py`) for training.
   - Train the model using SageMaker's `fit()` method.

5. **Deploy the Model**:
   - Deploy the trained model using the `deploy()` method to create a SageMaker endpoint.
   - Use the endpoint to make predictions.

6. **Evaluate the Model**:
   - Verify the predictions using a small batch of test data.
   - Analyze the classification report and accuracy score.

7. **Clean Up Resources**:
   - Delete the endpoint after testing to avoid unnecessary charges.
   - Retain the model artifacts in S3 for future use.

---

## Example Output

1. **Model Artifact**:
   - Stored in S3 after training.
   - Path: `s3://<bucket_name>/<prefix>/model.joblib`.

2. **Endpoint**:
   - A deployed endpoint for real-time predictions.
   - Example usage:
     ```python
     predictions = predictor.predict(testX[features][0:2].values.tolist())
     print(predictions)
     ```

3. **Test Metrics**:
   - Accuracy and classification report printed after evaluating the model.

---

## Clean Up

To avoid incurring unnecessary costs, make sure to:
- Delete the SageMaker endpoint:
  ```python
  sm_boto3.delete_endpoint(EndpointName=endpoint_name)
  ```
- Retain only essential resources like model artifacts and datasets in S3.

---

## Next Steps

1. Experiment with different machine learning models and hyperparameters.
2. Use **Amazon SageMaker Pipelines** to automate the workflow.
3. Integrate advanced features like **Model Monitoring** and **AutoML**.

