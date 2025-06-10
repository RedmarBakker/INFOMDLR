import numpy as np
import matplotlib.pyplot as plt
import mne

def plot_dataset_as_lines(data):
    # === Create a time axis ===
    sfreq = 2034  # Hz
    n_channels, n_samples = data.shape
    time = np.arange(n_samples) / sfreq  # in seconds

    # === Plot a few channels ===
    plt.figure(figsize=(15, 6))
    for i in range(248):  # Plot first 5 sensors
        plt.plot(time, data[i] * 1e15, label=f'Sensor {i} (offset)', linewidth=0.8)
        # Multiplied by 1e15 to convert to femtoTesla

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude + Offset (fT)')
    plt.title('MEG Sensor Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dataset_as_meg(data, frame=1):
    n_channels, n_times = data.shape

    sfreq = 2048
    ch_names = [f'CH{i}' for i in range(n_channels)]  # neutral names to avoid MNE rejecting them
    ch_types = ['eeg'] * n_channels  # initially pretend they are EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create nearly-flat (non-coplanar) positions in 3D
    np.random.seed(42)  # For reproducibility
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    radius = 1.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 0.01 * np.random.randn(n_channels)  # add tiny jitter in Z-axis

    positions = np.vstack((x, y, z)).T

    montage = mne.channels.make_dig_montage(
        {name: pos for name, pos in zip(ch_names, positions)},
        coord_frame='head'
    )

    info.set_montage(montage)
    info.set_channel_types({name: 'mag' for name in ch_names})

    first_frame = data[:, frame-1:frame]
    evoked = mne.EvokedArray(first_frame, info)
    evoked.plot_topomap(times=0, ch_type='mag', size=4.0 ,time_format='', units='fT')

    # fig = plt.figure(figsize=(2, 2))  # ~200x200 px
    # topo_ax = fig.add_axes([0, 0, 1, 1])  # full figure
    # topo_plot = None
    #
    # # === Animation update function ===
    # def update(frame_idx):
    #     global topo_plot
    #     topo_ax.clear()
    #     data_at_t = data[:, frame_idx].reshape(n_channels, 1)
    #     evoked.data = data_at_t  # update evoked data
    #     evoked.plot_topomap(times=0, ch_type='mag', size=4.0, time_format='', units='fT')
    #
    # ani = FuncAnimation(fig, update, frames=range(0, n_times, 10), interval=30)
    plt.show()