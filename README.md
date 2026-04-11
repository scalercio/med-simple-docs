# med-simple-docs

This repository contains code and resources used to support the annotation and automatic simplification of Portuguese drug leaflets in the healthcare domain.

## Overview

This project was developed in the context of research on **Automatic Text Simplification (ATS)** for medical documents, with a focus on **drug leaflets written in Portuguese**.

The repository includes:

- Code used to **support human annotation and revision** of simplified drug leaflets  
- Code used to **generate automatic simplifications** using Large Language Models (LLMs)  
- Data resources used in the experiments described in the associated paper  

## Data

### Simplified Drug Leaflets Dataset

The repository contains a dataset of **30 drug leaflets for hypertension medications**, which were:

- Automatically simplified using LLMs  
- Revised by non-linguists  
- Further revised following linguistic guidelines  

📂 Location: data/bulas_all_v2.parquet


This file includes aligned pairs of:
- Original drug leaflet texts  
- Simplified and revised versions  

### Original Drug Leaflets

The original PDF leaflets used in the study are available in:

📂 Folder: hipertensao/


These documents were collected from different manufacturers and used as the source material for simplification.

## Purpose

This repository supports research on:

- Simplification of **safety-critical medical information**  
- Evaluation of **readability vs. content preservation trade-offs**  
- Development of **annotation protocols for healthcare NLP**  

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{scalercio2026annotation,
  title={Annotation Guidelines and Challenges for Automatic Simplification of Portuguese Drug Leaflets},
  author={Scalercio, Arthur and Bertotto, Eduarda and Jesus, Silvana and Finatto, Maria Jos{\'e} and Paes, Aline},
  booktitle={Proceedings of the International Conference on Computational Processing of the Portuguese Language (PROPOR)},
  year={2026}
}
