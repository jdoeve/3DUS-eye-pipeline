# uceye

Modular UC-Eye radial ultrasound pipeline.

## API

```python
from uceye import run_pipeline
```

## CLI

```bash
run_uceye --input-dir /path/to/frames --cone-mask /path/to/cone_mask.png --out-dir /path/to/output
```

## Dev

```bash
cd /Users/jeffreydoeve/Desktop/3DUS-Eye/uceye_package
python3 -m pip install -e .
pytest
```
