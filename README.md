# Quick Start Guide - CP219 Project 2 with UV

## Step 7: Set Up W&B (Optional but Recommended)

```bash
# Login to W&B
wandb login

# Enter your API key from: https://wandb.ai/authorize

# Update config.yaml with your team name
```

## Step 8: Run Tasks

### Run Task 1 (EDA)

```bash
# Copy task1_eda.py to src/tasks/
# Then run:
python -m src.tasks.task1_eda
```

Or create a simple runner in `experiments/`:

```bash
# Create experiments/run_task1.py
cat > experiments/run_task1.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.tasks.task1_eda import run_task1

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = run_task1(config, logger=None)
    print("\nTask 1 completed!")
EOF

# Run it
python experiments/run_task1.py
```

### Run Task 2 (Feature Characterization)

```bash
# Create experiments/run_task2.py
cat > experiments/run_task2.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.tasks.task2_characterization import run_task2

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = run_task2(config, logger=None)
    print("\nTask 2 completed!")
EOF

# Run it
python experiments/run_task2.py
```

### Run with W&B

```python
# experiments/run_task1_wandb.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.tasks.task1_eda import run_task1
from src.utils.wandb_utils import WandbLogger

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize W&B
    logger = WandbLogger(
        project_name=config['project']['name'],
        entity=config['project'].get('entity'),
        config=config,
        job_type='task1_eda',
        tags=['task1', 'eda']
    )

    # Run task
    results = run_task1(config, logger)

    # Finish
    logger.finish()
    print("\nTask 1 completed with W&B logging!")
```

## Step 9: Check Outputs

```bash
# View generated figures
ls outputs/figures/task1/
ls outputs/figures/task2/

# View generated tables
ls outputs/tables/task1/
ls outputs/tables/task2/

# Open figures (macOS)
open outputs/figures/task1/*.png

# Open figures (Linux)
xdg-open outputs/figures/task1/*.png

# Open figures (Windows)
start outputs/figures/task1/*.png
```

## Step 10: Team Collaboration

### Each Team Member:

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd cp219_project2
   ```

2. **Set up UV environment**

   ```bash
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   uv pip install -e ".[dev]"
   ```

3. **Login to W&B**

   ```bash
   wandb login
   ```

4. **Run experiments with your name tag**

   ```python
   logger = WandbLogger(
       project_name="cp219-goose-ids",
       entity="your-team-name",
       tags=['task1', 'alice']  # Add your name
   )
   ```

5. **View results in W&B dashboard**
   - Go to https://wandb.ai/your-team-name/cp219-goose-ids
   - Compare runs across team members
   - Share insights

## Common Commands

```bash
# Install new package
uv pip install package-name

# Update dependencies
uv pip install --upgrade package-name

# List installed packages
uv pip list

# Freeze dependencies (for requirements.txt)
uv pip freeze > requirements.txt

# Run tests
pytest tests/

# Format code
black src/

# Lint code
flake8 src/

# Start Jupyter notebook
jupyter notebook notebooks/
```

## Troubleshooting

### Issue: Module not found

```bash
# Make sure you're in the project root and activated venv
pwd  # Should show project directory
which python  # Should show .venv/bin/python

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: UV command not found

```bash
# Add UV to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Issue: Data files not found

```bash
# Check data directory
ls data/raw/

# Ensure files are named correctly (case-sensitive!)
# Should match names in config.yaml
```

### Issue: W&B not logging

```bash
# Re-login
wandb login --relogin

# Check W&B status
wandb status

# Run with debug mode
WANDB_DEBUG=true python experiments/run_task1.py
```

## Next Steps

1. ✅ Complete Task 1 (EDA)
2. ✅ Complete Task 2 (Feature Characterization)
3. ⬜ Implement Task 3 (Binary Detection)
4. ⬜ Implement Task 4 (Multi-class Detection)
5. ⬜ Optional: Task 5 (Advanced Analyses)
6. ⬜ Write report
7. ⬜ Prepare submission

## Resources

- **UV Documentation**: https://docs.astral.sh/uv/
- **W&B Documentation**: https://docs.wandb.ai/
- **Project Specification**: CP219_Project2_2025.pdf
- **Team Dashboard**: https://wandb.ai/HPR-cp219 /cp219-goose-ids

---

**Happy Coding! 🚀**
