# %%

# Import the Quality Tools from Python Quality Assurance Module
from pyFOQA import Quality
import os

# Set the base directory to QAR JSON Data
dir = 'B737'
ac = '683'
# dataset = '683200404051447.json'
dataset = None

# Create the Quality object
path = os.path.join(dir, ac)
q1 = Quality(path, dataset)

# %%

# Run the timeSeries Tool

timelimits = None
q1.time_series('EGT_1', time_limits = timelimits)
q1.shewhart_chart('FF_1', 'PH', 'EGT_1')
q1.correlation('FF_1', 'FF_2')

# %%
