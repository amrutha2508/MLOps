# deploying models as a service

1. create a virtual env with pipenv
2. create a script for predicting
3. Putting the script into a Flask App
4. Packaging the app to Docker

```bash
docker build -t ride-duration-prediction-service:v1 .
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```
