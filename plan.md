# Plan: Fine-Tuning gemma4 e2b for Housing Assistant

## Context
The goal is to transform the base `gemma4 e2b` model into a specialized chatbot assistant for the housing domain. The model needs to understand specific terminology (e.g., "prefarea", "furnishingstatus") and the relations between property attributes (e.g., how area and amenities influence price). 

This is the first phase of a two-step project:
1. **Fine-Tuning**: Teach the model the "language" and "logic" of the housing data using `Housing.csv`.
2. **RAG Implementation (Future)**: Integrate a secondary knowledge source via Retrieval Augmented Generation, where the fine-tuned model will act as the expert interpreter of the retrieved data.

The intended outcome is a model that can conversationally explain housing data patterns and accurately interpret property specifications without hallucinating unrealistic attributes.

## Implementation Approach

### 1. Data Preprocessing
Transform the tabular `Housing.csv` into a conversational instruction-tuning dataset (`.jsonl` format).

- **Synthetic Instruction Generation**: Create a script `src/preprocess.py` to generate three types of training pairs:
    - **Descriptive**: "Describe property X" $\rightarrow$ "Property X is a [furnished] house with [4] bedrooms..."
    - **Relational**: "How does [furnishingstatus] affect price?" $\rightarrow$ "Generally, fully furnished homes command a higher price..."
    - **Interpretive**: "Given [Data Snippet], is this a luxury home?" $\rightarrow$ "Yes, because it has [prefarea=yes] and [area > 5000]..."
- **Formatting**: Use the Gemma chat template: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>`.

### 2. Model Setup & Training
Use Parameter-Efficient Fine-Tuning (PEFT) to optimize for performance and resource usage.

- **Technique**: **QLoRA (4-bit Quantized LoRA)**.
- **Critical Files**:
    - `src/train.py`: Main training loop using `trl.SFTTrainer`.
    - `configs/train_config.yaml`: Hyperparameters (Learning Rate: $2 \times 10^{-4}$, Epochs: 3-5, LoRA Rank: 16).
- **Target Modules**: Fine-tune `q_proj`, `k_proj`, `v_proj`, and `o_proj` layers.
- **RAG Readiness**: Train the model to reason over provided snippets rather than just memorizing the CSV, ensuring it can later handle dynamic RAG contexts.

### 3. Verification & Evaluation
Compare the fine-tuned model against the base `gemma4 e2b` model.

- **A/B Testing**: Use `src/eval.py` to run 50 domain-specific queries across both models.
- **Metrics**:
    - **Terminology Accuracy**: Check for correct use of "prefarea" and "furnishingstatus".
    - **Relational Logic**: Verify if the model correctly associates larger areas/better furnishing with higher price brackets.
    - **LLM-as-a-Judge**: Use a larger model (e.g., Gemma 27B) to rate the naturalness and accuracy of responses.

## Critical Files
- `/home/azapto/project/fineTune/src/preprocess.py` (New)
- `/home/azapto/project/fineTune/src/train.py` (New)
- `/home/azapto/project/fineTune/src/eval.py` (New)
- `/home/azapto/project/fineTune/configs/train_config.yaml` (New)
- `/home/azapto/project/fineTune/data/Housing.csv` (Existing)

## Verification Plan
1. **Preprocessing Check**: Verify `train.jsonl` contains a diverse set of prompt/response pairs and follows the Gemma chat template.
2. **Training Convergence**: Monitor loss curves during training to ensure the model is learning without overfitting.
3. **End-to-End Test**:
    - Ask the model to describe a specific property from the test set.
    - Ask a relational question (e.g., "What makes a house expensive in this dataset?").
    - Verify that the fine-tuned model's answers are more domain-accurate than the base model's.
