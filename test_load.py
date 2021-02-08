import load
from pathlib import Path

base_path = f'{Path.home()}/SageMaker'
print(base_path)

g_ema = load.generator(base_path).cuda()

