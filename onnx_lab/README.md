# Lab: ONNX Model Optimization and Inference

## Objective
Export a trained scikit-learn model to ONNX format, run inference using ONNX Runtime,
and benchmark performance against the native scikit-learn model.

## Execution Steps
1. Install dependencies:
   pip install scikit-learn pandas onnx onnxruntime skl2onnx

2. Train and export model:
   python train_and_export.py

3. Compare inference performance:
   python infer_compare.py
