# Deep Learning Exercises — FAU WS23

> Deep learning exercise solutions from FAU Erlangen-Nürnberg Winter Semester 2023. Covers neural network fundamentals through CNNs and advanced architectures.

## Architecture

**Stack**: Python + PyTorch + Jupyter Notebooks

```
exercise_0/          → Introduction / Setup
exercise_1/          → Neural network basics
exercise_2/          → Backpropagation and training
exercise_3/          → CNNs and advanced topics
exercise_4/          → Advanced architectures
Logos/               → Course logos and branding
```

## Developer Workflows

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # if available per exercise

# Run notebooks
jupyter notebook

# Run Python scripts
python exercise_X/solution.py
```

## Conventions

- **Exercises are self-contained** — each directory has its own notebooks/scripts
- **PyTorch**: Use `torch.nn.Module` subclassing pattern for models
- **Notebooks**: Include theory explanations in markdown cells, code in code cells
- **Data**: Downloaded datasets go to exercise-specific `data/` dirs (gitignored)

## Gotchas

1. **GPU recommended** — some exercises require significant compute for training
2. **PyTorch versions** — exercises were written for specific PyTorch version; check per-exercise requirements
3. **Exercise dependencies** — later exercises may build on earlier ones
4. **Logos directory** — course branding assets, don't modify
