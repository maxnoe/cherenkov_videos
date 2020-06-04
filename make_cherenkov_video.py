import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from argparse import ArgumentParser
import os

from eventio import IACTFile
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('indir')
parser.add_argument('outputfile')

masses = {
    'Electrons': 0.000511,
    'Muons': 0.105658,
    'Other': None,
}

FPS = 50
FRAMES_PER_NS = 12.5
N_BINS = 500
RADIUS = 500


inputfiles = [
    'gamma_1TeV.eventio.zst',
    'proton_1TeV.eventio.zst',
    'helium_1TeV.eventio.zst',
    'iron_10TeV.eventio.zst',
]

primaries = {
    1: 'Gamma',
    14: 'Proton',
    402: 'Helium',
    5626: 'Iron',
}

DURATION = {'gamma_1TeV.eventio.zst': 20}
MIN = {
    'gamma_1TeV.eventio.zst': 0.25,
    'iron_10TeV.eventio.zst': 0.5,
}


class CherenkovAni:
    def __init__(self, events):
        self.events = events

        self.total_frames = sum(e['frames'] for e in events)
        self.bar = tqdm(total=self.total_frames)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='k')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_aspect(1)
        ax.set_facecolor('k')
        ax.set_axis_off()

        self.img = ax.imshow(
            np.zeros((N_BINS, N_BINS, 3)),
            extent=[-RADIUS, RADIUS, -RADIUS, RADIUS]
        )

        self.title_text = ax.text(
            0.05, 0.95, '',
            color='white', va='top', ha='left',
            size=32, transform=ax.transAxes, family='Fira Mono',
        )
        ax.text(
            0.95, 0.05, '[M. Nöthe]',
            color='lightgray', va='bottom', ha='right',
            size=24, transform=ax.transAxes, family='Fira Mono',
        )
        self.time_text = ax.text(
            0.95, 0.95, '', color='white', size=48, family='Fira Mono',
            transform=ax.transAxes, va='top', ha='right',
        )

        ax.plot([-RADIUS + 100, -RADIUS + 200], [-RADIUS + 100, -RADIUS + 100], 'w', lw=4)
        ax.text(
            -RADIUS + 150, -RADIUS + 105, '100 m', color='white', size=24,
            family='Fira Mono', va='bottom', ha='center'
        )

        self.title_fmt = '{} {:.0f} TeV \nR: e⁺/e⁻\nG: µ⁺/µ⁻\nB: other\n'
        self.fig = fig
        self.ax = ax
        self.current = 0
        self.frame = 0

    def init(self):
        return self.img, self.time_text, self.title_text

    def update(self, i):
        self.bar.update(1)

        if self.frame >= self.events[self.current]['frames']:
            print('next event')
            self.current += 1
            self.frame = 0

        event = self.events[self.current]

        image = event['images'][self.frame]
        image = image.astype(float) / event['scale']
        image = np.clip(image, 0, 1)
        self.img.set_array(image)
        text = '{:3.0f} ns'.format(event['time'][self.frame] - event['t_min'])
        self.time_text.set_text(text)

        self.frame += 1
        if self.frame == 1:
            self.title_text.set_text(
                self.title_fmt.format(event['primary'], event['energy'])
            )
            return self.img, self.time_text, self.title_text

        return self.img, self.time_text


def read_event(inputfile):
    name = os.path.basename(inputfile)
    print('Reading', name)
    f = IACTFile(inputfile)
    event = next(iter(f))
    print('Reading done')
    print(f'RADIUS: {RADIUS} m')

    photons = event.photon_bunches[0]
    mask = (np.sqrt(photons['x']**2 + photons['y']**2) / 100) < RADIUS
    photons = photons[mask]
    time = photons['time']
    time -= time.min()
    primary = primaries[event.header['particle_id']]
    energy = event.header['total_energy'] / 1e3

    emitter = event.emitter[0][mask]

    t_min = np.percentile(time, MIN.get(name, 1))
    if name not in DURATION:
        t_max = np.percentile(time, 90)
    else:
        t_max = t_min + DURATION[name]
    frames = int(FRAMES_PER_NS * (t_max - t_min))

    print(f'Time range: {t_min:.2f} {t_max:.2f} ns')
    print(f'Frames: {frames}')

    images = np.zeros((frames, N_BINS, N_BINS, 3), dtype='uint32')
    for channel, (particle, mass) in enumerate(masses.items()):
        if mass is not None:
            mask = np.isclose(emitter['mass'], mass, rtol=1e-4)
        else:
            mask = emitter['mass'] > 0.106

        hist, edges = np.histogramdd(
            np.column_stack([
                time[mask],
                photons['x'][mask] / 100,
                photons['y'][mask] / 100,
            ]),
            bins=[frames, N_BINS, N_BINS],
            range=[[t_min, t_max], [-RADIUS, RADIUS], [-RADIUS, RADIUS]],
        )
        images[:, :, :, channel] = hist

    time = 0.5 * (edges[0][1:] + edges[0][:-1])
    scale = np.percentile(images[images > 0], 99.5)
    return dict(
        time=time,
        scale=scale,
        images=images,
        primary=primary,
        t_min=t_min,
        energy=energy,
        frames=frames,
    )


def main():
    args = parser.parse_args()

    events = []
    for inputfile in inputfiles:
        events.append(read_event(os.path.join(args.indir, inputfile)))

    c = CherenkovAni(events=events)

    ani = FuncAnimation(
        c.fig, c.update, init_func=c.init,
        frames=c.total_frames, interval=1000 / FPS,
        blit=True,
    )
    ani.save(args.outputfile, savefig_kwargs=dict(facecolor='k'))


if __name__ == '__main__':
    main()
