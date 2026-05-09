# MLOps

1. mlruns directory(artifact-store) - anything that is heavy like model dataset preprocessor images etc any artifact
    1. mlflow.log_artifact()
2. mlflow.db(backend-store) - stores all the metadata regarding all the runs like name parameters metrics anything that you log , which will be light weight.
3. tracking server - 
    1. no tracking server - mlflow ui
    2. local tracking server - mlflow server

