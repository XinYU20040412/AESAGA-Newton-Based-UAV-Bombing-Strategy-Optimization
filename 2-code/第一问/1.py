import numpy as np
from cover_checker import *
from system_at_t import *
time=0
checker = AdvancedMissileSmokeChecker()
cover_system_pro1=cover_system([120],[np.pi],np.array([[1.5]]),np.array([[5.1]]))
for t in np.arange(0, 67, 0.1):
    Mj,smokes_location=cover_system_pro1(t,1)
    if checker.check(Mj, smokes_location):
        time+=0.1
        print(time)



