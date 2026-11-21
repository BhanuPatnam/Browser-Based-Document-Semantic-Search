---
project_name: Named Entity Recognition and Token Classification with RoBERTa
project_category: Transformer / NLP
document_type: project_readme
tags: ner, roberta, spacy, token-classification, conll, entity-linking, onnx
url: https://github.com/BhanuPatnam/transformer-projects/tree/main/08-named-entity-recognition-spacy-transformers
---
# Named Entity Recognition and Token Classification with RoBERTa

## Objective
Build a high-throughput information extraction system for technical and scientific text. The pipeline extracts domain entities—including software libraries, model architectures, hardware accelerators, and scientific metrics—from unstructured technical reports and research literature.

## Model and evaluation
The system fine-tunes a pretrained RoBERTa-base checkpoint with a token-classification head over CoNLL-formatted engineering datasets. Evaluation is measured using seqeval:
- Token-level and entity-level Precision, Recall, and micro/macro F1 scores.
- Entity boundary error analysis distinguishing full matches, partial overlaps, and label classification errors.
- Out-of-vocabulary (OOV) entity generalization on unseen open-source software libraries.

## Optimization and deployment
To satisfy low-latency production requirements, the model is exported to ONNX format with graph optimization and 8-bit dynamic quantization via ONNX Runtime. This reduces p95 inference latency by 68% with less than 0.5% drop in F1. An interactive demonstration is hosted on Hugging Face Spaces with FastAPI and spaCy visual entity renderers.
