import mlflow

def start_experiment(name):

    mlflow.set_experiment(name)

    mlflow.start_run()

def log_metric(name, value):

    mlflow.log_metric(name, value)

def end_run():

    mlflow.end_run()