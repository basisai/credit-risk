# credit-risk

### Data
Data is adapted from [here](https://www.kaggle.com/c/home-credit-default-risk/data).

### Use `ModelAnalyzer`
See [notebook](doc_modelanalyzer.ipynb)

### Run on Bedrock
Parameters will passed to your scripts as environment variables. They can be overwritten when you create a pipeline run.

### Test your server
```
curl -X POST \
  <MODEL_ENDPOINT_URL> \
  -H 'Content-Type: application/json' \
  -H 'X-Bedrock-Api-Token: <MODEL_ENDPOINT_TOKEN>' \
  -d '{"sk_id": "302427"}'
```

### Credit risk analysis prototype
[App](https://boiling-badlands-89141.herokuapp.com/)
