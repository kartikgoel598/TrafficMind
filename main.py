import os
import sys


# this part of the code below is only needed due to school computer blocking setting SUMO as PATH
# add to where sumo-1.26.0 is in your computer
SUMO_HOME = r"D:\Users\270487801\Downloads\sumo-win64-1.26.0\sumo-1.26.0"
os.environ["SUMO_HOME"] = SUMO_HOME


sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci

sumoBinary = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")

print('starting simulation... (main.py)')
print(SUMO_HOME)

traci.start([
    sumoBinary,
    "-c",
    r"D:\Users\270487801\Desktop\TrafficMind\sumo\config\basic.sumocfg"
])