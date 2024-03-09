# llama2_emotion
Obtain parameters that can be used for emotional reasoning through LoRA

This parameter is mainly trained based on the three datasets of [HuggingFace](https://huggingface.co)'s [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) and [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion). 
It can be used to analyze, identify and summarize the emotions contained in the text.

The parameters are as follows:

- **avg_train_prep**, Value: 1.3647297620773315
- **avg_train_loss**, Value: 0.31094954411188763
- **avg_eval_prep**, Value: 2.0137341022491455
- **avg_eval_loss**, Value: 0.6999893188476562
- **avg_epoch_time**, Value: 11819.33716046686
- **avg_checkpoint_time**, Value: 0.7591855948170027

Quick Start Guide:
==================
## Download this project：
```
git clone https://github.com/YKHC/llama2_emotion.git
```
## Navigate to directory
```
cd llama2_emotion
```
## When you load the model you can add this sentence in your code：
```
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
#model_id = "/Llama-2-7b-hf"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, "/emotion_analysis")
```
## Your prompt should be in this format:
```
prompt = "Analyze the sentiment of this sentence:\n{}\n---\nSentiment:\n"
```
