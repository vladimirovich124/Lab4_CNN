@echo off
echo Установка зависимостей...
pip install -r requirements.txt

echo Запуск MLflow UI...
start mlflow ui

echo Запуск пайплайна DVC...
dvc repro
