### TODO

1. Establish the "Locked" Evaluation Set (High Priority)
   Objective: Create the 100-problem benchmark before any training occurs to avoid "cheating."

[x] Harvest 150 potential questions:

[x] Extract 50 from Science Olympiad (human-verified).

[x] Extract 50 from RealFP (human-verified).

[x] Generate 50 new "Cross-Domain" questions using Claude/GPT.

[ ] Perform Manual Verification:

[ ] For every question, find the "Ground Truth" constants via WolframAlpha or Wikipedia. especially on fermi_problems_synth_1.json

[ ] Manually calculate the Fermi Exponent using the Science Olympiad "Rule of Five."

[ ] Filter and Finalize:

[ ] Select the 100 highest-quality, unambiguous problems.

[ ] Format them into a eval_set.json file with fields for question, reasoning_steps, and gold_exponent.

[ ] Ensure Data Separation: - [ ] Run a string-similarity check to ensure NONE of these 100 questions exist in your training sources. \*\*\* important

2. Training Data Preparation & Filtering

[ ] Write a prompt for a frontier model to act as a classifier: "Does this problem require scientific domain knowledge (Bio/Chem/Phys/Earth)? Answer YES/NO."

[ ] Execute the Ablation Split:

[ ] Subset A (The "Everything" Set): All valid problems (~13k).

[ ] Subset B (The "Science Only" Set): Only problems tagged "YES" by your filter (~3k-5k).

[ ] Synthesize Reasoning Traces:

[ ] For problems that only have an answer but no steps (like the Science Olympiad set), use a frontier model to generate a Chain-of-Thought (CoT) that leads to that answer.

3. Establish Scaffolding & Baseline
   Objective: Determine how much the "brain" can do versus the "tools."

[ ] Standardize the Prompt: - [ ] Finalize your TELeR Level 3 system prompt (Clear instructions + Step-by-step enforcement). Not really necessary anymore.

[ ] Run the Baseline:

[ ] Benchmark your base 8B model (e.g., Llama 3.1 8B) on the 100-problem Eval set.

[ ] Record scores for two modes: No-Calculator (pure reasoning) vs. Python-Augmented (calculator use).

4. Training Infrastructure (Tinker/Fine-tuning)
   Objective: Set up the environment for SFT and RL.

[ ] SFT Setup: - [ ] Prepare your training script using LoRA or QLoRA to fit the 8B model on your available hardware.

[ ] Reward Function Implementation: - [ ] Write the Python script for the RL Reward Signal that calculates fp_score based on the difference between the model's output exponent and the gold exponent.

[ ] Run Stage 1 (SFT): - [ ] Train on Subset B (Science Only) and measure if accuracy on the Eval set improves over the baseline.

5. Documentation & Transparency
   Objective: Address the professor's feedback on methodology.

[ ] Log Prompting Strategies: Create a table documenting the exact prompts used for baseline vs. training.

[ ] Verify Bias: Check if the model performs significantly better on Claude-generated questions vs. Science Olympiad questions.

### Questions to Answer

-   How do we give the LLM context to the questions?
-   We also need to generate new questions?
-
