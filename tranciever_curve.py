from ocsim import QamSignal,SignalSetting
from ocsim import load_ase
from ocsim import LmsPll
from ocsim import scatterplot
from ocsim import PulseShaping
from ocsim import snr_meter
from ocsim import WSS
import numpy as np

import tqdm
snrs = []

for tranciever_noise in [26,25.7,25.4,25.1]:
    snr_one_tranciever = []

    for osnr in tqdm.tqdm(np.arange(15,45,1/2)):
        signal = QamSignal(SignalSetting(193.1e12))

        shaping = PulseShaping(0.2)
        signal = shaping(signal)
        # wss = WSS(0, 37.5e9, 8.8e9)
        # for i in range(5):
        #     signal = wss(signal)
        signal = load_ase(signal,tranciever_noise)

        signal_ase = load_ase(signal,osnr)

        signal_ase = shaping(signal_ase)
        signal_ase.normalize()
        lmspll = LmsPll(77,0.,0.001,3,3,lr_dd=0.)
        signal_ase = lmspll(signal_ase)

        snr_one_tranciever.append(snr_meter(signal_ase))
    snrs.append(np.array(snr_one_tranciever))

from FigureManger import *

data = DataSetting(x = np.vstack([np.arange(15,45,1/2)] * 4), y = np.vstack(snrs))
layout = Layout(x_axis_name="OSNR [dB]",y_axis_name="SNR [dB]",legend=('Tranciever 1','Tranciever 2','Tranciever 3','Tranciever 4'),
                markers=("*",'o','>','^'),style=('science','ieee','grid','no-latex')
                )
fig = FigureManger(data,layout,keyword=('Tranciever 1','Tranciever 2','Tranciever 3','Tranciever 4'))
fig.plot()