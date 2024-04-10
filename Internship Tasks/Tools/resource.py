import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button


class PlotAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.x = np.linspace(-10, 10, 1000)
        self.N = 200
        self.interv = 50
        self.n0 = (
            1.0
            / (4 * np.pi * 2e-4 * 0.1) ** 0.5
            * np.exp(-self.x**2 / (4 * 2e-4 * 0.1))
        )
        (self.p,) = self.ax.plot(self.x, self.n0)
        self.anim_running = True
        self.Myanimation = animation.FuncAnimation(
            self.fig, self.update, frames=self.N, interval=self.interv, blit=True
        )

    def update(self, i):
        self.n0 += i / 100 % 5
        self.p.set_ydata(self.n0 % 20)
        return (self.p,)

    def animate(self):
        pause_ax = self.fig.add_axes((0.7, 0.025, 0.1, 0.04))
        pause_button = Button(pause_ax, "pause", hovercolor="0.975")
        pause_button.on_clicked(self._pause)
        plt.show()

    def _pause(self, event):
        if self.anim_running:
            self.Myanimation.event_source.stop()
            self.p.set_animated(False)
            self.anim_running = False
            self.fig.canvas.draw_idle()
        else:
            self.p.set_animated(True)
            self.Myanimation.event_source.start()
            self.anim_running = True
