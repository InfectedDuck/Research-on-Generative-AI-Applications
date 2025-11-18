# Generative AI Vocabulary (SME cheat sheet)

This file lists specific terms you should know and use when imagining yourself as a subject-matter expert in generative-AI development and deployment.

1. LLM (Large Language Model)
   - A neural network trained on very large text corpora to predict tokens and generate text; used for completion, summarization, and conversational tasks.
   - Example use: "We evaluated three LLMs for latency and hallucination rate."

2. Tokenization
   - Breaking text into tokens (subwords/words) that the model processes; influences model input length and cost.
   - Example: "Switching tokenizers changed effective sequence length and throughput."

3. Fine-tuning
   - Continued training of a pre-trained model on task-specific data to improve behavior or align with domain style.
   - Example: "We fine-tuned the base model on our internal docs to reduce hallucinations."

4. Prompt engineering
   - Crafting input prompts to guide LLM outputs; includes templates, few-shot examples, and system messages.
   - Example: "Prompt engineering reduced off-topic replies in user testing."

5. Hallucination
   - When a generative model produces incorrect or fabricated facts that are not grounded in source data.
   - Example: "A hallucination in the answer made us add an explicit retrieval step."

6. Retrieval-Augmented Generation (RAG)
   - An architecture that retrieves documents (via search or vector DB) and conditions the LLM on them to improve factuality.
   - Example: "We implemented RAG for enterprise Q&A to ground answers in indexed policy documents."

7. Embedding
   - A numeric vector representing semantic content of text used for similarity search and clustering.
   - Example: "We used embeddings to build our semantic search index."

8. Few-shot / Zero-shot learning
   - Few-shot: model is given a small number of examples in the prompt; zero-shot: no examples, just instructions.
   - Example: "Few-shot prompts improved performance on domain-specific templates."

9. Multimodal
   - Models or systems that handle multiple data types (e.g., text + images + audio).
   - Example: "A multimodal assistant can answer a question about a screenshot."

10. Diffusion models
    - A class of generative models for continuous data (commonly used for images) that transform noise into samples via iterative denoising.
    - Example: "Stable Diffusion is a diffusion-based image generator."

11. Transfer learning
    - Reusing representations learned on one task/domain to accelerate training or improve performance on another.
    - Example: "We used transfer learning from a general corpus before domain-specific fine-tuning."

12. Grounding
    - The process of connecting generated answers to external sources or data (to reduce hallucination).
    - Example: "Grounding answers in the database reduced unsupported claims."

13. Inference
    - Running a trained model to generate outputs given new inputs (can be online or batch).
    - Example: "Inference cost was the main expense in production."

14. Adversarial robustness
    - The model’s resistance to crafted inputs that cause incorrect or harmful outputs.
    - Example: "We added input validation to improve adversarial robustness."

15. Data provenance
    - Tracking the origin and lineage of training or retrieved data used to produce model outputs.
    - Example: "We store provenance metadata for every document returned by the retriever."

Notes:
- Manually learning these terms and using them in explanations demonstrates SME-level familiarity and will help you earn extra points on the assignment.
