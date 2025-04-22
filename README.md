# 🍽️ Food Image Captioning Pipeline
This project implements a pipeline for generating, refining, and evaluating captions for food images using vision-language models. It combines the strengths of InstructBLIP, CLIP, and LLaMA to produce human-aligned, high-quality descriptions in domain-specific settings like food imagery.

## 🚀Pipeline Overview
1. Baseline Caption Generation
Model: Salesforce/instructblip-flan-t5-xl

Script: scripts/baseline.py

Output: captions_baseline.json

Goal: Generate concise, neutral food image captions (zero-shot, no context).

2. Scoring with CLIP & SLA
Models: CLIP (ViT-L/14)

Script:

scripts/image_embeddings.py

scripts/caption_embeddings.py

scripts/clip_sla_score.py

Outputs:

*_clip_scores.json

*_sla_scores_from_embeddings.json

Goal: Evaluate captions based on visual-textual alignment and syntax fluency.

3. Caption Refinement (Optional)
Models: LLaMA-2-7B-chat or GPT-4

Script: scripts/refining_captions.py

Output: refined_captions.json

Goal: Improve weak captions using LLMs while preserving factual visual content.

4. Fine-tuning InstructBLIP
Script: finetuning_experiments/<experiment_name>/finetuning.py

Approach: Fine-tune decoder layers using LoRA.

Output: instructblip_finetuned/

Goal: Improve model alignment and descriptiveness for food domain.

5. Evaluation
Script: scripts/evaluation.py

Tasks:

Re-caption test split using fine-tuned models.

Score with CLIP & SLA.

Evaluate with BLEU, METEOR, CIDEr on datasets with GT (e.g., Food500Cap).

## 📊 Metrics

Metric	Description
CLIP Score	Cosine similarity between image and caption embeddings.
SLA Rank	Caption's rank based on cosine similarity (lower is better).
BLEU/ROUGE/METEOR/CIDEr	Used on Food500Cap (which has ground-truth captions).
📁 Scripts & Key Files

## 📁 Scripts & Key Files
scripts/baseline.py	Generate zero-shot InstructBLIP captions
scripts/image_embeddings.py	Compute CLIP image embeddings
scripts/caption_embeddings.py	Compute CLIP caption embeddings
scripts/clip_sla_score.py	Compute CLIPScore and SLA
scripts/refining_captions.py	Refine captions with LLM
scripts/evaluation.py	Evaluate metrics (BLEU, ROUGE, etc.)
finetuning_experiments/*/	Fine-tuning experiments & results

## 📂 Datasets

- **[Food101](https://www.kaggle.com/datasets/dansbecker/food-101)** – Used for pseudo-labeling & zero-shot experiments.
- **[Food500Cap (Hugging Face)](https://huggingface.co/datasets/advancedcv/Food500Cap/viewer/default/train)** – Rich, curated food captions used for training & evaluation.

## 🧠 Models Used
Caption Generator: Salesforce/instructblip-flan-t5-xl
Caption Scoring:	openai/clip-vit-large-patch14
Caption Refinement:	meta-llama/Llama-2-7b-chat-hf / GPT-4

## 🛠️ Environment Requirements
Python 3.9+

PyTorch (CUDA enabled)

HuggingFace transformers

openai-clip, tqdm, Pillow, LAVIS

## 📌 Notes
Modular pipeline: change filenames or datasets to reuse scripts.

Use caption_log.txt for tracking generated samples.

Baseline and fine-tuned results are separated for easy comparison.

Experiments are organized under finetuning_experiments/.

## 📍How to Run (Summary)

# 1. Generate Baseline Captions
python scripts/baseline.py

# 2. Generate Embeddings & Score
python scripts/image_embeddings.py
python scripts/caption_embeddings.py
python scripts/clip_scores.py
python scripts/sla_scores.py
python scripts/clip+sla.py

# 3. Refine Captions (Optional)
python scripts/refining_captions.py

# 4. Fine-tune InstructBLIP
python finetuning_experiments/experiment_x/finetuning.py

# 5. Generate New Captions + Evaluate
python scripts/generate_captions_test.py
python scripts/evaluation.py

## 🤝 Credits
Developed as part of the CV703 Vision-Language Research Project @ MBZUAI.
Project by: Amal Saqib, Karina Abubakirova, Khawla Ali Hasan Ali Almarzooqi, Noora Al Hajeri
