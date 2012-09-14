#!/usr/bin/python
import fabio, feature, time
print feature.__file__
ref0 = 1.6923630028440242
ref1 = 0.67084262245579773
img1 = fabio.open("Ti_slow_data_0000_0011_0000_norm.edf").data
img2 = fabio.open("Ti_slow_data_0002_0055_0000_norm.edf").data
#out = feature.sift2(img1, img2 , True)
#assert  abs((out[:, 2] - out[:, 0]).mean() - ref0) < 0.01
#assert  abs((out[:, 3] - out[:, 1]).mean() - ref1) < 0.01

import threading
dico = {}
sift = feature.SiftAlignment()
def oneImg(name):
    img1 = fabio.open(name).data
    dico[name] = sift.sift(img1)
th1 = threading.Thread(target=oneImg, args=["Ti_slow_data_0000_0011_0000_norm.edf"])
th2 = threading.Thread(target=oneImg, args=["Ti_slow_data_0002_0055_0000_norm.edf"])
t0 = time.time()
th1.start();th2.start()
th1.join();th2.join()

print time.time() - t0, dico
t0 = time.time()
out = sift.match(dico["Ti_slow_data_0000_0011_0000_norm.edf"], dico["Ti_slow_data_0002_0055_0000_norm.edf"])
print time.time() - t0
assert  abs((out[:, 2] - out[:, 0]).mean() - ref0) < 0.01
assert  abs((out[:, 3] - out[:, 1]).mean() - ref1) < 0.01

