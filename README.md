# Full Waveform Inversion Project

This is a deep learning-based full waveform inversion project for seismic velocity model inversion.

## Project Structure

```
.
├── configs/          # Configuration files
├── data/            # Data directory
├── guided_diffusion/ # Guided diffusion model related code
├── models/          # Model files
├── scripts/         # Script files
├── storage/         # Storage directory
├── test_data/       # Test data
├── util/            # Utility functions
├── create_v.py      # Velocity model generation script
├── plot.py          # Plotting tools
├── repeat.py        # Repeated sampling script
├── repeat.sh        # Repeated sampling shell script
├── run_sampling.sh  # Sampling script
└── requirements.txt # Project dependencies
```

## Requirements

- Python 3.x
- CUDA support (recommended for GPU acceleration)
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/geguojia/ODE-DPS
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model files:
   Due to file size limitations, model files are not included in the repository.
   Please download the following files and place them in their respective directories:

   - models/CurveFault-B.pt
   - models/CurveVel-A.pt
   - models/CurveVel-B.pt
   - util/model/ffhq_10m.pt

   Download link: contact geguojia@sjtu.edu.cn

## Usage

1. Configure experiment parameters:
   - Modify experiment settings in `configs/forward_process.yaml`

2. Generate velocity model:
   - Use `create_v.py` to modify velocity model parameters

3. Run sampling:
   - Single sampling: run `run_sampling.sh`
   - Multiple sampling: run `repeat.sh`

4. Analyze results:
   - Use `plot.py` for visualization
   - Use `statistic.py` for statistical analysis

## Main Features

- Full waveform inversion based on guided diffusion model
- Customizable velocity model generation
- Batch sampling and repeated experiment capabilities
- Result visualization and statistical analysis tools

## Notes

- Ensure sufficient GPU memory for model training and inference
- Sampling process may take considerable time
- Recommended to run small-scale tests before large experiments
