
from ocsim import QamSignal,SignalSetting

signal = QamSignal(SignalSetting(device='cuda',center_freq=193.1e12))
print(signal.device)