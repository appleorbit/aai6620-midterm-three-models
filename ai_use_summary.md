# AI Use Summary

I used Claude (made by Anthropic) throughout this project as a tutor and coding assistant. I didn't use any other AI tools. This file explains what the AI did, what I did myself, and what I understand.

## How AI Helped

**Planning.** It helped me break the assignment into steps and suggested using Google Colab, the Tiny Shakespeare dataset, and character-level modeling.

**Code.** It assisted me in writing code, provided ideas, and gave code explanations to help me understand, such as the data pipeline, shared training/evaluation/generation functions, the three models, the supplementary experiment, the results table, the loss plots, and the attention visualizations in the main notebook. I wrote the code for the core functions. I ran all the code in Colab and fully understood the intent of the code and the running results.

**Explanations.** It explained in detail the meaning of each part of the code and how they relate to each other. For parts I found difficult to understand, it explained them in my native language, Chinese, such as how Q, K, and V work, why the bigram model plateaus, what residual connections do, how to read overfitting from the loss curves, and how the causal mask works.

**Writing.** I was the primary driver of the content and wrote the outline and framework. In the notebook, I not only organized all the analytical ideas in Chinese, but also completely rewrote each section based on the actual experimental results; afterward, I used AI to translate my Chinese writing into English and had it help polish some sentence structures, and I also used AI to proofread my drafts and provide revision suggestions. The final reflection section was completed independently by me. For the README, I used the AI-generated framework draft as a skeleton, but in terms of the core content, reflections, and alignment with the results, everything went through my own significant modifications and strict verification to ensure that what is presented reflects my true experimental conclusions.

**Experiment ideas.** It suggested the 15,000 step supplementary run and the two-speaker attention test, and it pointed out the dropout difference between the LSTM and the Transformer. I decided to include all three.

**Debugging and checking.** It explained the errors I ran into and cross-checked every number in the write-up against the actual outputs. Along the way it caught and corrected a few of its own mistakes.

**Grammar.** It corrected the grammar and word choice in the English I wrote myself.

## What I Did Myself

**Running the project.** I set up the GitHub repository and the Colab GPU environment and ran every experiment, about 25 minutes of training in total. When my Colab session reset and wiped all the variables, I reran everything and recovered the results. I organized the notebook structure and uploaded all the files to the repository.

**Checking the work.** I sent the output of every step back for review. When I wasn't sure the screenshots were clear enough, I downloaded the full-resolution figures and had the analysis rechecked, which made the attention descriptions more accurate.

**Making decisions.** I decided to run the extra experiments, to add the dropout caveat, and how to name the files and repository.

**Learning the material.** Before keeping any analysis, I asked for a Chinese translation so I could make sure I understood it. I also asked follow-up questions about the Head 2 attention pattern and about why the LSTM overfit more.

**Reimplementing the core code.** In a separate notebook (`aai6620_practice_reimplementation.ipynb`), I rewrote every core component of the project from scratch, with the AI guiding me through the steps: the data pipeline, the bigram model, the attention head, multi-head attention, the feedforward layer, the Transformer block, the full Transformer model, the LSTM model, the training and evaluation functions, and text generation. I tested each one as I went, checking output shapes and parameter counts, and fixed the problems the AI pointed out, such as an interface inconsistency in my first bigram version. I then trained all three models with my own code in a short run to confirm the pieces work together: the losses went down and all three generated text. I added a note in that notebook explaining that this run used a shorter context and isn't meant to match the main results.

**Answering concept questions.** I explained in my own words what happens without the causal mask, why attention scores are scaled by √d, why the bigram loss stops improving, why the random starting point in `get_batch` has to stop at `len(data) - block_size`, why `register_buffer` is used for the mask, how the parameter counts of the attention, feedforward, and block layers add up, why the Transformer needs positional embeddings but the LSTM doesn't, why `model.eval()` and `model.train()` matter during evaluation, and why generation samples from the distribution instead of always picking the most likely character.

**Writing in my own words.** Besides rewriting the notebook's text, the paragraph in the attention section about getting stuck on "why negative infinity instead of 0" and the reflection at the end of my practice notebook come from my own experience writing and running the code.

## What I Understand

I can explain how each of the three models works, why the bigram hits a ceiling, why the target in `get_batch` is the input shifted by one character, how to read overfitting from the train and validation curves, and what the attention heatmaps show. The causal mask was the part I didn't get at first. After the AI explained it, I could explain in my own words what it does and why removing it would make the training loss look perfect while the generated text falls apart.

## How I Used AI Responsibly

Every number in this project comes from runs I did myself, and nothing in the results was made up or estimated. When the analysis made a claim I could test, like what Head 2 was doing, I ran an experiment to check it instead of just accepting the first explanation. I treated the AI as a tutor: I used its explanations to learn the material, rebuilt the core code on my own, and made sure I could follow and agree with everything before it went into the final submission.
