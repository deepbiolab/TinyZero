# TinyZero
![image](cover.png)

TinyZero is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown and multiplication tasks. We built upon [veRL](https://github.com/volcengine/verl).

Through RL, the 3B base LM develops self-verification and search abilities all on its own 

You can experience the Ahah moment yourself for < $30 

Twitter thread: https://x.com/jiayi_pirate/status/1882839370505621655

Full experiment log: https://wandb.ai/jiayipan/TinyZero

Paper's on it's way!


## Update: Quick start (For those who are interested on running it on 2 H100s/A100s

This is a condensed and runnable guide for setting up and running **TinyZero** on [Hyperbolic](https://app.hyperbolic.xyz/) using the specified environment.

**1. Set up [Hyperbolic](https://app.hyperbolic.xyz/) H100 Environment**
1. **Sign up/sign in:**
   - [W&B](https://wandb.ai)
   - [Hyperbolic](https://app.hyperbolic.xyz/)

2. **Start a machine:**
   - **Instance:** A100 SXM or H100 SXM, select 2 GPUs.
   - **Image:**  
     `nvidia-cuda124-ubuntu2204`

**2. System Preparation**
Run the following commands to update and install dependencies:

```bash
# Update and upgrade system
sudo apt update && sudo apt upgrade -y

# Install necessary packages
sudo apt install -y git python3 python3-pip

# create virtual env and activate 
python -m venv myenv
source myenv/bin/activate
```

**3. Clone and Set Up TinyZero**
```bash
# Clone the TinyZero repository
git clone https://github.com/JerryWu-code/TinyZero.git

# Navigate into the TinyZero directory
cd TinyZero
```

**4. Install Dependencies**
```bash
# Install PyTorch (or let vLLM handle the correct version)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip3 install vllm==0.6.3  # Other versions: 0.5.4, 0.4.2, 0.3.1

# Install Ray
pip3 install ray

# Install TinyZero in editable mode
pip install -e .

# Install Flash Attention 2
pip3 install flash-attn --no-build-isolation

# Install quality-of-life tools
pip install wandb IPython matplotlib
```

**5. Log in to W&B**
```bash
wandb login
```

**6. Download Dataset**
```bash
huggingface-cli download Jiayi-Pan/Countdown-Tasks-3to4 \
  --local-dir ./data/countdown --repo-type dataset
```

**7. Preprocess Dataset**
```bash
python ./examples/data_preprocess/countdown.py --local_dir ./data/countdown
```

---

#### **8. Download and Save Pretrained Model**
```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'Qwen/Qwen2.5-3B'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for Flash Attention 2.0
)
model.save_pretrained('/home/ubuntu/TinyZero/models/Qwen2.5-3B')
tokenizer.save_pretrained('/home/ubuntu/TinyZero/models/Qwen2.5-3B')
"
```

**9. Set Environment Variables**
```bash
export N_GPUS=2
export BASE_MODEL="./models/Qwen2.5-3B"
export DATA_DIR="./data/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
```

**10. Update Training Script**
Edit the training script to remove unnecessary prefixes from the Python path:
```bash
vim scripts/train_tiny_zero_a100_grpo.sh
```

- Remove `/home/weiji/anaconda3/envs/zero/bin/` from the first line so it directly references `python3`.

**11. Make Script Executable**
```bash
chmod +x ./scripts/train_tiny_zero_a100_grpo.sh
```

**12. Start Training**
```bash
./scripts/train_tiny_zero_a100_grpo.sh
```

-----


## Original Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## Countdown task

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

### Run Training
```
conda activate zero
```

For the following code, if you see Out-of-vram, try add `critic.model.enable_gradient_checkpointing=True` to the script, and checkout the discussion [here](https://github.com/Jiayi-Pan/TinyZero/issues/5#issuecomment-2624161643)

**Single GPU**


Works for model <= 1.5B. For Qwen2.5-0.5B base, we know it fails to learn reasoning.

```
export N_GPUS=1
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

**3B+ model**
In this case, the base model is able to develop sophisticated reasoning skills.
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

### Instruct Ablation
We experiment with QWen-2.5-3B Instruct too.
**Data Preparation**
To follow chat template, we need to reprocess the data:
```
conda activate zero
python examples/data_preprocess/countdown.py --template_type=qwen-instruct --local_dir={path_to_your_dataset}
```

**Training**
```
export N_GPUS=2
export BASE_MODEL={path_to_your_model}
export DATA_DIR={path_to_your_dataset}
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-instruct
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh
```

## Acknowledge
* We run our experiments based on [veRL](https://github.com/volcengine/verl).
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).

## Citation
```
@misc{tinyzero,
author       = {Jiayi Pan and Junjie Zhang and Xingyao Wang and Lifan Yuan and Hao Peng and Alane Suhr},
title        = {TinyZero},
howpublished = {https://github.com/Jiayi-Pan/TinyZero},
note         = {Accessed: 2025-01-24},
year         = {2025}
}
```
