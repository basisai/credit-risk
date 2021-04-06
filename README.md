# credit-risk

## Goals
At the end of the tutorial, the user will be able to
- set up a Bedrock training pipeline
- log training-time feature and inference distributions
- log model explainability and fairness metrics
- check model explainability and fairness from Bedrock web UI
- deploy a model endpoint in HTTPS with logging inference and feature distributions
- monitor the endpoint by simulating a query stream

### Data
The data can be downloaded from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data). We have already uploaded the dataset on GCS and AWS.

### Use `ModelAnalyzer`
You can refer to the [notebook](doc_modelanalyzer.ipynb) for an overview of [Bedrock ModelAnalyzer](https://docs.basis-ai.com/guides/explainability-and-fairness).

## Run on Bedrock
Just follow the Bedrock quickstart guide from [Step 2](https://docs.basis-ai.com/guides/quickstart/step-2-create-a-project) onwards. You can either test on Google Cloud or AWS by setting `gcp` or `aws` in the **ENV_TYPE** field on the **Run pipeline** page.

## Check model performance, explainability and fairness
After successful run of training pipeline, clicking on the corresponding "Model version" will bring you to the "Model" page. You can select between "Metrics", "Explainability", "Fairness" and "File listing". You can also download the model artefacts saved during training.

In [Explainability](https://docs.basis-ai.com/guides/explainability-and-fairness/explainability#step-6-visualise-on-bedrock-ui), you will be able to visualise top feature attributions for the model at a global level as well as the SHAP dependence for selected features. You can view individual explainability by selecting the row index from the sampled dataset.

In [Fairness](https://docs.basis-ai.com/guides/explainability-and-fairness/fairness#step-6-visualise-on-bedrock-ui), you can visualise the fairness metrics on the Bedrock UI. You can select the protected attribute from the dropdown menu.

## Test your server
You can simulate a constant stream of queries with `query_stream.py`. Replace "MODEL_ENDPOINT_URL" and "MODEL_ENDPOINT_TOKEN" in `query_stream.py`. Run
```
python query_stream.py
```

## Monitor the endpoint
On the "Endpoint" page, you can select between "API metrics", "Feature distribution" and "Inference distribution".

From API metrics, you can monitor the throughput, response time and error rates.

From [Feature distribution](https://docs.basis-ai.com/guides/detect-feature-drift#step-3-real-time-analysis), you can compare training-time and production-time distributions for each feature in the form of CDF and PDF plots. Note that the production plots will only appear after 15 minutes.

Similarly, from [Inference distribution](https://docs.basis-ai.com/guides/customising-model-monitoring/collecting-inference-metrics), you can compare training-time and production-time distributions of inferences.

## Credit risk analysis prototype
The [Streamlit app](https://boiling-badlands-89141.herokuapp.com/) provides a demonstration of credit risk analysis.
