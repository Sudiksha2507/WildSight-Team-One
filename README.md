# WildSight – AI-Powered Wildlife Monitoring & Camera-Trap Analytics

WildSight is a Deep Learning-based wildlife monitoring system that uses computer vision to analyse camera-trap images, detect animals, classify supported species, and present wildlife activity through an interactive dashboard.

The project explores the application of deep learning techniques to real-world wildlife monitoring, with a focus on animal detection, image classification, transfer learning, model evaluation, and confidence-based predictions.

## Table of Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Deep Learning Concepts Applied](#deep-learning-concepts-applied)
* [System Architecture](#system-architecture)
* [Key Features](#key-features)
* [Technology Stack](#technology-stack)
* [Dataset](#dataset)
* [How It Works](#how-it-works)
* [Model Evaluation](#model-evaluation)
* [Results](#results)
* [Dashboard](#dashboard)
* [Installation and Setup](#installation-and-setup)
* [Limitations](#limitations)
* [Future Scope](#future-scope)
* [References](#references)

## Overview

Camera traps are widely used by researchers to monitor wildlife in natural environments. However, manually analysing thousands of camera-trap images is time-consuming and requires considerable effort.

WildSight addresses this challenge through a computer vision pipeline that automatically detects animals, predicts supported species, and organises wildlife observations for further analysis.

The system combines a pretrained animal detection model with a fine-tuned deep learning classification model and an interactive dashboard.

## Problem Statement

Large camera-trap datasets contain thousands of images, including animal photographs, empty frames, and images containing humans or vehicles.

Manually identifying animals and analysing their appearances across different locations and time periods is difficult.

The objective is to develop a system that automates important parts of this process using deep learning and presents the resulting information in an understandable format.

## Objectives

1. Detect animals in camera-trap images and filter empty images.
2. Identify supported animal species using a deep learning classification model.
3. Display prediction confidence and handle uncertain classifications.
4. Analyse wildlife sightings based on species, location, and time.
5. Present results through an interactive dashboard with charts and representative images.
6. Explore the practical application and limitations of deep learning in wildlife monitoring.

## Deep Learning Concepts Applied

WildSight demonstrates several important concepts from Deep Learning and Computer Vision.

### 1. Convolutional Neural Networks (CNNs)

CNNs are deep learning architectures commonly used for image-processing tasks.

They automatically learn visual features from images, such as:

* Edges and textures.
* Shapes and patterns.
* More complex features representing objects and species.

CNN-based architectures are particularly useful for recognising animals because they can learn distinguishing visual patterns from image data.

### 2. Object Detection

Object detection involves identifying objects present in an image and locating them using bounding boxes.

WildSight uses **MegaDetector V6** through PyTorch-Wildlife to identify animals, people, and vehicles in camera-trap images.

The detection stage helps remove irrelevant or empty images before species classification.

### 3. Image Classification

Image classification assigns an image to one of the supported categories.

WildSight uses a fine-tuned **MobileNetV3-Small** model to classify detected animal images into 13 supported species classes.

The model produces predictions along with confidence scores.

### 4. Transfer Learning

Transfer learning uses knowledge learned by a model on an existing dataset for a new task.

Instead of training a deep neural network entirely from scratch, a pretrained model can be adapted to a specific classification problem.

WildSight uses MobileNetV3-Small as the starting point for its species classification task.

### 5. Fine-Tuning

Fine-tuning involves adapting a pretrained model to a target dataset by training it for the new task.

In WildSight, MobileNetV3-Small was fine-tuned for wildlife species classification over 5 epochs.

This approach makes use of previously learned visual features while adapting the model to the supported wildlife categories.

### 6. Confidence-Based Prediction

A classification model produces confidence scores for its predicted classes.

WildSight uses a confidence threshold of **0.60** to identify uncertain predictions.

Predictions below this threshold are marked as **Unidentified**, rather than being treated as reliable species identifications.

This helps reduce the risk of presenting uncertain classifications as confirmed results.

### 7. Model Evaluation

A deep learning model must be evaluated using appropriate performance metrics.

WildSight uses the following metrics:

* **Accuracy:** Measures the proportion of correct predictions.
* **Precision:** Measures how many predicted instances of a class are correct.
* **Recall:** Measures how many actual instances of a class are identified correctly.
* **F1-score:** Combines precision and recall using their harmonic mean.
* **Macro F1-score:** Calculates the F1-score independently for each class and averages the results.
* **Weighted F1-score:** Calculates the average F1-score while considering the number of samples in each class.

These metrics provide a better understanding of classification performance than accuracy alone.

### 8. Handling Uncertainty and Generalisation

Wildlife images may vary because of lighting, camera angles, backgrounds, animal positions, and environmental conditions.

A model that performs well on one set of images may not perform equally well on images from different camera locations.

WildSight uses confidence filtering to identify uncertain predictions. Its evaluation results should also be interpreted with the limitations of the validation split in mind.

## System Architecture

The project follows a multi-stage processing pipeline:

```text
       Camera-Trap Dataset
                |
                v
       Image Preprocessing
                |
                v
       MegaDetector V6
                |
                v
      Animal Detection
                |
                v
       Empty Image Filtering
                |
                v
     Detected Animal Images
                |
                v
   MobileNetV3-Small Classifier
                |
                v
       Species Prediction
                |
                v
      Confidence Threshold
                |
        +-------+-------+
        |               |
        v               v
   Identified       Unidentified
     Species         Predictions
        |               |
        +-------+-------+
                |
                v
       Sightings Generation
                |
                v
       Wildlife Analytics
                |
                v
       Interactive Dashboard
```

## Key Features

### Animal Detection

Uses MegaDetector V6 to identify animals, people, and vehicles in camera-trap images.

### Species Classification

Uses a fine-tuned MobileNetV3-Small model to classify detected animal images into 13 supported classes.

### Confidence Filtering

Displays prediction confidence and marks predictions below the 0.60 threshold as unidentified.

### Sightings Analysis

Groups related detections using camera sequence information or timestamp proximity to organise wildlife observations.

### Interactive Dashboard

Displays wildlife statistics, species distribution, location-based information, time-based trends, and representative images.

### Wildlife Insights

Integrates the Featherless API for natural-language summaries based on aggregated wildlife statistics.

## Technology Stack

| Category             | Technologies                      |
| -------------------- | --------------------------------- |
| Programming Language | Python, TypeScript                |
| Deep Learning        | PyTorch                           |
| Object Detection     | MegaDetector V6, PyTorch-Wildlife |
| Image Classification | MobileNetV3-Small                 |
| Computer Vision      | OpenCV                            |
| Data Processing      | NumPy, Pandas                     |
| Frontend             | Next.js, React, Tailwind CSS      |
| UI Components        | shadcn/ui                         |
| API Integration      | Featherless API                   |
| Dataset              | Caltech Camera Traps              |

## Dataset

WildSight uses the **Caltech Camera Traps dataset**, accessed through the LILA (Labeled Information Library of Alexandria) platform.

The dataset contains camera-trap images collected from different locations and environmental conditions.

For this project, a subset of 5,000 images was selected for processing.

Dataset characteristics include:

* Images collected from multiple camera locations.
* Different animal species and environmental conditions.
* Empty frames and images containing non-animal objects.
* Image annotations used for detection and analysis.

The original dataset and downloaded image files are not included in this repository.

## How It Works

The project processes camera-trap images through the following stages:

1. **Dataset Preparation:** A subset of camera-trap images is selected and downloaded.
2. **Animal Detection:** MegaDetector V6 identifies animals, people, and vehicles.
3. **Image Filtering:** Images without relevant animal detections are filtered out.
4. **Species Classification:** Detected animal images are passed to the fine-tuned MobileNetV3-Small classifier.
5. **Confidence Evaluation:** Predictions are checked against the confidence threshold.
6. **Sightings Generation:** Related detections are organised using camera sequence information or timestamp proximity.
7. **Dashboard Visualisation:** Processed wildlife information is presented through charts, statistics, and representative images.

## Model Evaluation

The species classification model was evaluated using a validation split.

The following metrics were obtained:

| Metric               | Result |
| -------------------- | -----: |
| Validation Accuracy  | 64.61% |
| Macro F1-score       |   0.56 |
| Weighted F1-score    |   0.64 |
| Confidence Threshold |   0.60 |
| Training Epochs      |      5 |

The validation split contained 243 samples.

The evaluation uses a random validation split rather than a held-out-camera split. Therefore, the results should not be interpreted as proof of generalisation to completely unseen camera locations.

## Results

The processed dataset produced the following results:

| Parameter                       | Result |
| ------------------------------- | -----: |
| Images Selected                 |  5,000 |
| Images Successfully Downloaded  |  3,914 |
| Camera Locations                |    140 |
| Animal Detections               |  6,305 |
| Predictions Marked Unidentified |  2,098 |

A confidence threshold of 0.60 was used to mark uncertain species predictions as unidentified.

These results demonstrate the end-to-end application of animal detection, species classification, confidence filtering, and wildlife data analysis.

## Dashboard

The interactive dashboard provides a visual representation of the processed wildlife data.

It includes:

* Overall wildlife statistics.
* Species distribution.
* Location-based sightings.
* Time-based activity analysis.
* Representative camera-trap images.
* Wildlife summaries based on aggregated statistics.

**Live Demo:** https://wildsight-eta.vercel.app/

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sudiksha2507/WildSight-Team-One.git
cd WildSight-Team-One
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

Navigate to the dashboard directory:

```bash
cd wildlife-dashboard
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the application at:

```text
http://localhost:3000
```

### 5. Environment Variables

Create a `.env.local` file inside the dashboard directory and configure the required API settings:

```env
FEATHERLESS_API_KEY=your_key_here
FEATHERLESS_MODEL=your_model_name_here
```

Replace the placeholder values with valid credentials and model information.

**Note:** Dataset images, trained model weights, and environment files are excluded from this repository. A fresh clone provides the project source code but does not automatically reproduce the complete trained deep learning pipeline or its results.

## Limitations

* The classifier supports 13 species classes rather than every possible animal species.
* Classification performance depends on image quality, lighting, camera position, and dataset representation.
* Predictions with confidence below 0.60 are marked as unidentified.
* The validation results are based on a random split of 243 samples, not a held-out-camera evaluation.
* The model may not generalise equally well to wildlife images from unseen locations.
* The dataset subset used in the project does not represent every possible wildlife environment.
* Reproducing the complete pipeline requires the dataset and trained model weights, which are not included in the repository.

## Future Scope

The project can be extended in several ways:

* Improve classification accuracy by training with a larger and more diverse dataset.
* Experiment with deeper CNN architectures and other pretrained models.
* Explore data augmentation to improve robustness to different lighting and environmental conditions.
* Evaluate generalisation using camera-wise or location-wise data splits.
* Extend the classifier to support additional animal species.
* Explore object tracking to analyse repeated animal appearances.
* Develop more advanced wildlife activity and population trend analysis.
* Improve uncertainty estimation and confidence calibration.
* Extend the dashboard with additional conservation-related insights.

## References

* [Caltech Camera Traps Dataset – LILA](https://lila.science/datasets/caltech-camera-traps/)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [PyTorch-Wildlife](https://github.com/microsoft/CameraTraps)
* [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
* [MegaDetector](https://github.com/agentmorris/MegaDetector)

## Project Information

**Project:** WildSight – AI-Powered Wildlife Monitoring & Camera-Trap Analytics

**Subject:** Deep Learning

**Repository:** https://github.com/Sudiksha2507/WildSight-Team-One

**Live Dashboard:** https://wildsight-eta.vercel.app/
