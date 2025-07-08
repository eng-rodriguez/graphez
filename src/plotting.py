

def plot_eegdata(raw, duration=15, scalings=10e-5, time_format="clock"):
    """Plot continous eegdata recording"""
    raw.plot(duration=duration, scalings=scalings, show_scrollbars=False, time_format=time_format)


def plot_epochs(epochs, n_epochs, scalings=10e-5):
    """Plot epochs eegdata recording"""
    epochs.plot(n_epochs=n_epochs, scalings=scalings, show_scrollbars=False)
    

def plot_sources(raw, ica):
    """Plot independent components sources"""
    ica.plot_sources(raw, show_scrollbars=False)
