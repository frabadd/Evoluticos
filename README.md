# Lung Cancer Prediction Project

This project aims to develop a lung cancer prediction system using a dataset of patients with various characteristics. By utilizing artificial intelligence techniques, specifically evolutionary methods, we aim to generate a model that predicts whether a patient might be at risk of developing lung cancer.

## Problem Description

Lung cancer prediction is key in helping people understand their risk at a low cost, which can facilitate better decision-making based on their health status. This project uses a dataset with 16 attributes and 284 instances that describe different characteristics of the patients, such as age, habits (smoking, alcohol consumption), and symptoms (cough, difficulty breathing, etc.).

## Dataset

The dataset includes the following attributes:

- **Sex:** M (Male), F (Female)
- **Age:** Integer
- **Smoker:** Yes (1), No (0)
- **Yellow fingers:** Yes (1), No (0)
- **Anxiety:** Yes (1), No (0)
- **High blood pressure:** Yes (1), No (0)
- **Chronic disease:** Yes (1), No (0)
- **Fatigue:** Yes (1), No (0)
- **Allergy:** Yes (1), No (0)
- **Wheezing:** Yes (1), No (0)
- **Alcohol consumption:** Yes (1), No (0)
- **Cough:** Yes (1), No (0)
- **Difficulty breathing:** Yes (1), No (0)
- **Difficulty swallowing:** Yes (1), No (0)
- **Chest pain:** Yes (1), No (0)
- **Lung cancer:** Yes (1), No (0) (Target label)

## Objective

The goal is to solve this problem as efficiently as possible using evolutionary methods. We will convert numerical values into binary ones to ease the data processing.

## Proposed Solutions

### Solution 1

A possible encoding for the model could be through a chromosome in base 3 [0, 1, *], where each gene corresponds to an attribute. The final solution will consist of several individuals in a population that collectively contribute to the success of the prediction.

### Solution 2

Another solution involves using binary chromosomes to indicate which attributes are considered, along with weight values for each attribute that determine its influence on the decision rule.

## Team

- Felipe Guzmán
- Álvaro Alcalde
- Rubén Ribes
- Odei Hijarrubia
- Francisco Prados
- Pablo Díaz-Masa

## Requirements

- Python 3.x
- Libraries:
  - Pandas
  - NumPy
  - Scikit-learn
