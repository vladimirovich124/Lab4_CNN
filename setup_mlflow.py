import mlflow

def setup_mlflow(experiment_name="default_experiment"):
    mlflow.set_experiment(experiment_name)
    return mlflow

mlflow = setup_mlflow("cnn_experiment")
