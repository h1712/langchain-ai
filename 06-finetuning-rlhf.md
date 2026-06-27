# 6. Fine-Tuning & RLHF

## Why Fine-Tune?

Pre-trained LLMs have broad knowledge but may not:
- Follow specific output formats consistently
- Know domain-specific terminology or processes
- Match the desired tone or style
- Perform well on niche tasks

**Fine-tuning** adapts a pre-trained model to a specific task using additional training on curated data.

```
Pre-trained Model (general knowledge)
        ↓ Fine-tune on domain data
Specialized Model (your use case)
```

---

## Fine-Tuning vs RAG vs Prompt Engineering

| Approach | When to Use | Data Needed | Cost | Flexibility |
|---|---|---|---|---|
| **Prompt Engineering** | Quick iteration, no training data | 0 examples | Free | Most flexible |
| **RAG** | Knowledge changes frequently, need citations | Documents | Low (embedding cost) | Update anytime |
| **Fine-Tuning** | Need behavioral changes, consistent formatting | 100s–1000s examples | Medium (training cost) | Retrain to update |
| **Full Pre-Training** | Building a new foundation model | Billions of tokens | Very High | Least flexible |

### Decision Framework
```
Is the model's format/style wrong? → Fine-tune
Is the model missing knowledge?   → RAG
Is the model reasoning poorly?    → Better prompt (CoT)
All of the above?                 → Fine-tune + RAG
```

---

## Supervised Fine-Tuning (SFT)

The most straightforward approach — train the model on (input, desired_output) pairs.

### Data Format
```json
[
  {"messages": [
    {"role": "system", "content": "You are a Jira ticket classifier."},
    {"role": "user", "content": "My build is failing on arm64"},
    {"role": "assistant", "content": "{\"type\": \"Bug\", \"priority\": \"High\", \"component\": \"Build\"}"}
  ]},
  {"messages": [
    {"role": "system", "content": "You are a Jira ticket classifier."},
    {"role": "user", "content": "Add dark mode to the dashboard"},
    {"role": "assistant", "content": "{\"type\": \"Feature Request\", \"priority\": \"Medium\", \"component\": \"UI\"}"}
  ]}
]
```

### OpenAI Fine-Tuning API
```python
# 1. Upload training data
file = client.files.create(file=open("training.jsonl"), purpose="fine-tune")

# 2. Start fine-tuning
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo",  # Base model
    hyperparameters={"n_epochs": 3}
)

# 3. Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo:org:custom-name:id",  # Fine-tuned model ID
    messages=[{"role": "user", "content": "My API returns 500 errors"}]
)
```

### How Many Examples?
| Quality | Examples | Notes |
|---|---|---|
| Minimum | 10 | Only for very simple tasks |
| Good | 50–100 | Most classification/formatting tasks |
| Better | 500–1000 | Complex domain tasks |
| Best | 5000+ | Significant behavior change |

**OpenAI guidelines:** At least 50 examples, ideally 100+ for reliable improvement.

---

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all model parameters (expensive for large models). PEFT methods update only a small subset.

### LoRA (Low-Rank Adaptation) — Hu et al., 2021

**Key insight:** Weight updates during fine-tuning are low-rank. Instead of updating the full weight matrix W, learn two small matrices A and B:

```
W_new = W_original + A × B
where A ∈ R^(d×r), B ∈ R^(r×d), r << d
```

| Parameter | Full Fine-Tune | LoRA (r=16) |
|---|---|---|
| GPT-3 (175B) | 175B params | ~17M params (0.01%) |
| LLaMA-7B | 7B params | ~4M params (0.06%) |
| Memory | Multiple GPUs | Single GPU possible |
| Training time | Days | Hours |

**LoRA config example:**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,              # Rank — higher = more capacity, more compute
    lora_alpha=32,     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
)

model = get_peft_model(base_model, config)
# Only ~0.1% of parameters are trainable!
```

### QLoRA (Quantized LoRA)

LoRA + 4-bit quantization of the base model:
```
Base model: 4-bit quantized (reduces memory 4×)
LoRA adapters: 16-bit (small, full precision)
```

**Result:** Fine-tune a 65B parameter model on a single 48GB GPU.

### Adapter Methods Comparison

| Method | Parameters Updated | Memory | Quality |
|---|---|---|---|
| Full Fine-Tune | 100% | Highest | Best |
| LoRA | ~0.1% | Low | Near-best |
| QLoRA | ~0.1% (4-bit base) | Lowest | Good |
| Prefix Tuning | Prepend learnable tokens | Low | Good for NLU |
| IA³ | Scale activations | Very Low | Limited |

---

## RLHF (Reinforcement Learning from Human Feedback)

The technique that made ChatGPT significantly better than GPT-3.

### The Three-Step Process (Ouyang et al., 2022 — InstructGPT)

```
Step 1: Supervised Fine-Tuning (SFT)
        → Train on human-written ideal responses

Step 2: Reward Model Training
        → Humans rank model outputs → train a model to predict rankings

Step 3: RL Optimization (PPO)
        → Use reward model to optimize the LLM's policy
```

### Step 1: SFT
```
Prompt: "Explain quantum computing"
Human-written response: "Quantum computing uses quantum bits (qubits)..."
Train: Supervised learning on (prompt, ideal_response) pairs
```

### Step 2: Reward Model
```
Prompt: "Explain quantum computing"
Response A: "Quantum computing is a type of computation..." (ranked 1st by human)
Response B: "Computers use quantum mechanics..." (ranked 2nd)
Response C: "It's complicated..." (ranked 3rd)

Reward Model learns: R(prompt, response_A) > R(prompt, response_B) > R(prompt, response_C)
```

The reward model is typically a smaller LLM with a scalar output head that predicts human preference scores.

### Step 3: PPO (Proximal Policy Optimization)
```
Objective: Maximize reward while staying close to the SFT model

Loss = -E[R(prompt, response)] + β × KL(π_RL || π_SFT)

Where:
- R = reward model score
- KL = KL divergence (prevents drifting too far from SFT model)
- β = penalty coefficient (controls exploration vs stability)
```

**Why KL penalty?** Without it, the model might "hack" the reward model — generating outputs that score high on the reward model but are actually low quality (reward hacking).

---

## DPO (Direct Preference Optimization) — Rafailov et al., 2023

**Key insight:** RLHF is complex (3 models, unstable training). DPO simplifies it to a single supervised training step.

```
RLHF: SFT model → Reward Model → PPO → Final Model  (3 stages)
DPO:  SFT model → Direct optimization from preferences → Final Model  (1 stage)
```

### How DPO Works
Instead of training a separate reward model, DPO directly optimizes the policy using preference pairs:

```
(prompt, chosen_response, rejected_response)
```

**Loss function:**
```
L_DPO = -log σ(β × (log π(chosen|prompt)/π_ref(chosen|prompt) - log π(rejected|prompt)/π_ref(rejected|prompt)))
```

**In plain English:** Increase the probability of the chosen response and decrease the probability of the rejected response, relative to a reference model.

### DPO vs RLHF

| Aspect | RLHF | DPO |
|---|---|---|
| Complexity | 3 models (SFT + RM + PPO) | 1 model + reference |
| Stability | Unstable (RL training) | Stable (supervised) |
| Compute | High | Lower |
| Performance | Slightly better | Comparable |
| Data needed | Ranked outputs | Preferred/rejected pairs |

---

## Constitutional AI (RLAIF) — Anthropic

Used by Claude. Replace human feedback with AI feedback:

```
1. Generate response
2. Ask another LLM: "Does this response violate any of these principles?"
3. If yes, ask LLM to revise
4. Train on revised outputs
```

**Principles (constitution) examples:**
- "Please choose the response that is most helpful"
- "Please choose the response that is least harmful"
- "Please choose the response that is most honest"

**Advantage:** Scales better than human feedback (no human labelers needed).

---

## Evaluation of Fine-Tuned Models

### Automated Metrics
| Metric | What It Measures |
|---|---|
| **Perplexity** | How surprised the model is by the test set (lower = better) |
| **BLEU** | N-gram overlap with reference (translation/generation) |
| **ROUGE** | Recall-oriented overlap (summarization) |
| **F1** | For classification tasks |
| **MMLU** | Multi-task language understanding benchmark |

### Human Evaluation
- **Side-by-side comparison:** Show outputs from base vs fine-tuned model
- **Likert scale rating:** Rate helpfulness, accuracy, safety (1–5)
- **Win rate:** Percentage of times the fine-tuned model is preferred

### LLM-as-Judge
Use GPT-4 to evaluate outputs:
```
Given this prompt and two responses, which is better and why?
Prompt: [...]
Response A: [...]
Response B: [...]
```

---

## Practical Considerations

### When Fine-Tuning Goes Wrong

| Problem | Symptom | Fix |
|---|---|---|
| **Overfitting** | Perfect on training data, poor on new inputs | More diverse data, fewer epochs |
| **Catastrophic forgetting** | Loses general capabilities | Lower learning rate, shorter training |
| **Reward hacking** | High reward score but poor actual quality | Better reward model, KL penalty |
| **Data contamination** | Memorizes training examples verbatim | Deduplicate, add diversity |

### Cost Estimation (OpenAI Fine-Tuning)

```
Training: ~$0.008 / 1K tokens (GPT-3.5-turbo)
Usage:    ~$0.012 / 1K tokens (3× base model price)

Example: 1000 examples × 500 tokens × 3 epochs = 1.5M tokens
Cost: 1.5M / 1000 × $0.008 = $12
```

### Your TejaBot's Approach

TejaBot does NOT use fine-tuning. Instead:
- **RAG** provides domain knowledge (from Milvus)
- **Prompt engineering** controls behavior (system prompt with rules)
- **Feedback loop** improves retrieval ranking (acceptance scores)

This is the right approach when:
- Knowledge changes frequently (new Slack conversations daily)
- The model's general capabilities are sufficient
- You need transparent, citable answers (RAG provides sources)

---

## Quick-Recall Points

- Fine-tuning = adapt pre-trained model with task-specific data
- LoRA: only train ~0.1% of params, nearly same quality as full fine-tune
- QLoRA: LoRA + 4-bit quantization → fine-tune 65B models on single GPU
- RLHF: SFT → Reward Model → PPO (3 steps, what made ChatGPT work)
- DPO: simplified alternative to RLHF (1 step, comparable results)
- KL divergence penalty prevents reward hacking in RLHF
- RAG > Fine-tuning when knowledge changes frequently
- Fine-tuning > RAG when behavior/style needs to change
- Minimum: 50–100 examples for meaningful fine-tuning

---

## ⭐ Interview / Exam Q&A

**Q: When would you choose fine-tuning over RAG?**
> Fine-tune when you need to change the model's behavior — output format, writing style, domain-specific reasoning patterns. Use RAG when you need to add external knowledge that changes frequently. They're complementary: fine-tune for behavior, RAG for knowledge. Many production systems use both.

**Q: Explain LoRA in simple terms.**
> Instead of updating all 7 billion parameters of a model, LoRA adds two small matrices (say 7B × 16 and 16 × 7B) whose product approximates the weight update. Only these ~4 million new parameters are trained. The original model is frozen. At inference, the LoRA weights are added to the original weights with no extra latency.

**Q: What is the difference between RLHF and DPO?**
> RLHF is a 3-stage process: (1) supervised fine-tuning, (2) train a separate reward model on human preferences, (3) use PPO (reinforcement learning) to optimize the LLM against the reward model. DPO collapses steps 2 and 3 into a single supervised learning step that directly optimizes from preference pairs (chosen vs rejected). DPO is simpler, more stable, and achieves comparable results.

**Q: What is reward hacking in RLHF?**
> The model learns to exploit weaknesses in the reward model — generating outputs that score high on the reward model but are actually low quality to humans. Example: the reward model might give high scores to verbose responses, so the model becomes extremely verbose without adding substance. KL divergence penalty (staying close to SFT model) and reward model quality are the primary defenses.

**Q: How does your TejaBot's feedback differ from RLHF?**
> TejaBot's feedback is much simpler — binary +1/-1 scores that directly influence retrieval ranking (Learning-to-Rank), not model weights. RLHF uses ranked preferences to train a reward model that then fine-tunes the LLM's parameters. TejaBot doesn't modify the LLM at all — it modifies which stored answers are surfaced to the LLM. This is computationally cheaper and doesn't require model retraining.

**Q: What is catastrophic forgetting?**
> When fine-tuning causes the model to "forget" its pre-trained capabilities. For example, a model fine-tuned on medical data might lose its ability to write code. Mitigations: (1) low learning rate, (2) fewer epochs, (3) mix general data with fine-tuning data, (4) LoRA (keeps original weights frozen).

**Q: Can you fine-tune GPT-4?**
> As of 2024, OpenAI offers fine-tuning for GPT-3.5-turbo and GPT-4o-mini. Full GPT-4 fine-tuning is in limited access. For most use cases, fine-tuning GPT-3.5-turbo + good prompting matches or exceeds base GPT-4. Alternatively, use open models (LLaMA, Mistral) with LoRA for full control.
