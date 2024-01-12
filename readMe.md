# Usage
## Installation 
```
git clone git@github.com:ibrahimEls/DNNBenchmark.git
cd DNNBenchmark/
conda create -n python3_pytorch python=3.8
conda activate python3_pytorch
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install requirements.txt
```
## Running the Benchmark
```
python3 timing_mluncertentiy_model.py
```