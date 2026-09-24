from pathlib import Path
import compileall, subprocess, sys, re
root=Path(__file__).resolve().parents[1]
print('compileall:',compileall.compile_dir(str(root),quiet=1,rx=re.compile(r'\.venv|__pycache__|results')))
print('required files:')
for p in ['run.py','requirements.txt','config.yaml','app/main.py','core/pipeline.py','experiments/run_experiment.py']:
    print(p, (root/p).exists())
