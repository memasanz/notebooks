# Databricks notebook source
# MAGIC %md
# MAGIC ### Track experiment runs and deploy ML models with MLflow and Azure Machine Learning (preview)
# MAGIC 
# MAGIC https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow

# COMMAND ----------

import mlflow
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication

# COMMAND ----------

# MAGIC %md
# MAGIC Connect to Azure ML Workspace - you will be asked to log in

# COMMAND ----------

subscription_id = dbutils.secrets.get(scope="mlg-kv", key="subscriptionid")

# COMMAND ----------

config = {
    "subscription_id": subscription_id,
    "resource_group": "mlops-RG",
    "workspace_name": "mlops-AML-WS"
}


print(config)


# COMMAND ----------

tenant_id = dbutils.secrets.get(scope="mlg-kv", key="tenantid")

interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id, cloud="AzureUSGovernment")

ws = Workspace.get(name=config['workspace_name'],
                   subscription_id=config['subscription_id'],
                   resource_group= config['resource_group'], auth=interactive_auth)

# COMMAND ----------

# MAGIC %md
# MAGIC Create an Azure ML Experiment

# COMMAND ----------

from azureml.core.experiment import Experiment
 

experimentName  = 'BaseExperimentWithAzureMLIntegration'
mlflow.set_experiment(experimentName)

experimentName = 'BaseExperimentWithAzureMLIntegration'

#experimentName = user_name + "BaseExperimentWithAzureMLIntegration"
experiment = Experiment(ws, experimentName)

# COMMAND ----------

# MAGIC %md
# MAGIC MLflow Tracking with Azure Machine Learning lets you store the logged metrics and artifacts from your local runs into your Azure Machine Learning workspace.
# MAGIC 
# MAGIC **Note:** The tracking URI is valid up to an hour or less. If you restart your script after some idle time, use the get_mlflow_tracking_uri API to get a new URI.

# COMMAND ----------

print(ws.get_mlflow_tracking_uri())
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

# MAGIC %md
# MAGIC - Set the MLflow experiment name with set_experiment() and start your training run with start_run(). Then use log_metric() to activate the MLflow logging API and begin logging your training run metrics.
# MAGIC 
# MAGIC - Any run with MLflow Tracking code in it will have metrics logged automatically to the workspace.

# COMMAND ----------



# COMMAND ----------

import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    with mlflow.start_run(run_name='tester'):
      X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
      y = np.array([0, 0, 1, 1, 1, 0])
      lr = LogisticRegression()
      lr.fit(X, y)
      score = lr.score(X, y)
      print("Score: %s" % score)
      mlflow.log_metric("score", score)
      mlflow.sklearn.log_model(lr, "model2")
      
      reg_model_name = "SklearnLinearRegression"
      print("--")
      loaded_model = lr
      mlflow.sklearn.log_model(loaded_model, "sk_learn",serialization_format="cloudpickle",registered_model_name=reg_model_name)


      print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
      run_id = mlflow.active_run().info.run_uuid
      mlflow.end_run()

# COMMAND ----------

user_id = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
user_name = user_id.split('@')[0]
print(user_name)

# COMMAND ----------

display(dbutils.fs.ls('dbfs:/'))

# COMMAND ----------

# MAGIC %md
# MAGIC Register the model in Azure ML

# COMMAND ----------

model_name = user_name + "wine_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/model2", model_name)

# COMMAND ----------

model_uri = "runs:/" + run_id + "/model2"

# COMMAND ----------

import mlflow.azureml
model_name = user_name + '_model'
model_image, azure_model = mlflow.azureml.build_image(model_uri=model_uri, 
                                                      workspace=ws,
                                                      model_name=model_name,
                                                      image_name="model",
                                                      description="image for predicting wine quality",
                                                      synchronous=False)

# COMMAND ----------

model_image.wait_for_creation(show_output=True)

# COMMAND ----------

from azureml.core.compute import AksCompute, ComputeTarget
 
# Use the default configuration (you can also provide parameters to customize this)
prov_config = AksCompute.provisioning_configuration()
 
aks_cluster_name = user_name + "aks-cluster" 
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_cluster_name, 
                                  provisioning_configuration = prov_config)
 
# Wait for the create process to complete
aks_target.wait_for_completion(show_output = True)
print(aks_target.provisioning_state)
print(aks_target.provisioning_errors)
