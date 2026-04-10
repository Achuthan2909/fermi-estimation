# Proposal

## Capability

What is the scientific capability? What makes it interesting and challenging?

Fermi Problems also known as Order-of-Magnitude (OoM) reasoning, involves estimating complex physical quantities through rapid "back-of-the-envelope" calculations. These calculations for typical LLM's pose an interesting problem as they are grounded on physical intuition. LLM's typically lack this physical intuition, making certain proposed solutions implausible. Although LLM's possess vast declarative knowledge they often lack this sense of physical intuition to do these consistent checks as it tries to solve through a problem, resulting in a hallucination.

## Evidence of Frontier Failure

~5 concrete examples where a frontier model (Gemini, Claude, ...) gets the task wrong or performs poorly. Include the prompts and outputs.

#### To Keep In Mind When Creating These Problems

To prove "Frontier Failure," don't use the "Piano Tuners" or "Grains of sand on a beach" problems—those are in the training data. Instead, combine two unrelated domains. \* Example: "Estimate the total length of DNA in all the bacteria living in the cooling ponds of every nuclear power plant on Earth."

This forces the model to synthesize: [Number of Plants] → [Pond Volume] → [Bacterial Density] → [DNA Length per Cell]. Current models almost always break one of these links.

#### Examples of Fermi Problems

## Gemini

## ChatGPT

## LLaMa

## Eval Plan

How will you measure success? What's the input format, output format, scaffolding (e.g., tool use), and scoring method?

## Data Plan

Where will your training data come from? Rough estimate of size. Do you plan to do SFT, RL, scaffolding improvements etc?
