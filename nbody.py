from __future__ import annotations

import math
import pygame
import pygame_widgets as pw
import randomcolor
import sys
import matplotlib.pyplot as plt

from collections import deque
from pygame_widgets.slider import Slider
from pygame_widgets.button import Button


rand_color = randomcolor.RandomColor()

G = 1  # we set the gravitational constant to 1
eps = 0.1  # force softening constant epsilon

# TODO: use the leapfrog integration method!

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1200


class Body:
    def __init__(self, m: float, x: float, y: float, vx: float, vy: float) -> None:
        self.m = m
        self.r: int = (
            None  # gets set by System.set_r() method, used when drawing the Body with PyGame
        )
        self.color: str = (
            None  # gets set by System() constructor, stored in hex color code
        )
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.speed = math.sqrt(vx**2 + vy**2)

        # updated body position after time step
        self.newx = x
        self.newy = y
        self.newvx = vx
        self.newvy = vy

        # keep track of body's position and display its tail
        self.hist = deque()
        self.hist_update_freq: int = None  # gets set in apply_update
        self.internal_count = (
            0  # tells us after how many time-steps should we update the tail
        )
        self.hist_size = 20  # determines the size of the tail

    def set_hist_update_freq(self, dist_update: float) -> int:
        # attempts to create equal sized tails for all bodies
        if dist_update == 0:
            return math.inf
        return round(5e-2 / (dist_update))

    def update(self, other_bodies: list[Body], dt: float) -> None:
        for other_body in other_bodies:
            rel_x = other_body.x - self.x
            rel_y = other_body.y - self.y
            d = math.sqrt(rel_x**2 + rel_y**2)
            theta = (
                math.atan(rel_y / rel_x) if rel_x != 0 else 0
            )  # only outputs values between -pi/2 and pi/2

            # force softening with epsilon
            a = (G * other_body.m * d) / ((d**2 + eps**2) ** 1.5)

            ax = abs(a * math.cos(theta)) * (1 if other_body.x >= self.x else -1)
            self.newvx += ax * dt
            self.newx += self.vx * dt

            ay = abs(a * math.sin(theta)) * (1 if other_body.y >= self.y else -1)
            self.newvy += ay * dt
            self.newy += self.vy * dt

    def apply_update(self) -> None:
        dist_update = math.sqrt((self.x - self.newx) ** 2 + (self.y - self.newy) ** 2)

        self.internal_count += 1
        self.hist_update_freq = self.set_hist_update_freq(dist_update)

        if self.internal_count > self.hist_update_freq:
            self.internal_count = 0
            self.hist.appendleft((self.x, self.y))

            if len(self.hist) > self.hist_size:
                self.hist.pop()

        # apply the updated values
        self.x = self.newx
        self.y = self.newy
        self.vx = self.newvx
        self.vy = self.newvy
        self.speed = math.sqrt(self.vx**2 + self.vy**2)

    def distance_to(self, other_body: Body) -> float:
        # returns the distance between two bodies
        return math.sqrt((self.x - other_body.x) ** 2 + (self.y - other_body.y) ** 2)


class System:
    def __init__(self, bodies: list[Body], dt: float) -> None:
        self.bodies = bodies
        colors = rand_color.generate(count=len(bodies))
        for body, color in zip(self.bodies, colors):
            body.color = color # color setting done here

        self.dt = dt
        self.Us, self.Ks, self.Es, self.dts = [], [], [], [0]
        self.compute_energy()

        self.set_r()

    def set_r(self) -> None:
        """
        Sets the radius for each body (used when drawing on PyGame).
        Radius is based on the size of the mass.
        """
        min_m = min(self.bodies, key=lambda body: body.m).m
        for body in self.bodies:
            body.r = round(7 * (math.log(body.m / min_m, 1e3) + 1))

    def update(self) -> float:
        # we scale our time-step by the sqrt of the distance between the two closest bodies
        closest_dist = min(1, self.compute_closest_distance() ** 0.5)
        new_dt = self.dt * closest_dist

        for i in range(len(self.bodies)):
            self.bodies[i].update(
                self.bodies[:i] + self.bodies[i + 1 :], new_dt
            )

        for i in range(len(self.bodies)):
            self.bodies[i].apply_update()

        self.compute_energy()
        self.dts.append(self.dts[-1] + new_dt)
        return new_dt

    def compute_energy(self) -> None:
        # just updates the lists where we store the energies, use [-1] to access the current energy
        U = 0
        for i in range(len(self.bodies) - 1):
            for j in range(i + 1, len(self.bodies)):
                U += (-G * self.bodies[i].m * self.bodies[j].m) / self.bodies[
                    i
                ].distance_to(self.bodies[j])

        K = 0
        for body in self.bodies:
            K += 0.5 * body.m * (body.speed**2)

        self.Us.append(U)
        self.Ks.append(K)
        self.Es.append(U + K)

    def plot_energies(self, energy_plot_filename: str = None) -> None:
        N = len(self.dts)
        spacing = round(N // 1000)
        plt.plot(self.dts[::spacing], self.Us[::spacing], color="blue", label="Potential Energy")
        plt.plot(self.dts[::spacing], self.Ks[::spacing], color="red", label="Kinetic Energy")
        plt.plot(self.dts[::spacing], self.Es[::spacing], color="green", label="Total Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.title("Energy vs. Time Plot")
        plt.legend()

        if energy_plot_filename:
            plt.savefig(energy_plot_filename)
        
        plt.show()

    def compute_closest_distance(self) -> float:
        closest_dist = self.bodies[0].distance_to(self.bodies[1])
        for i in range(len(self.bodies) - 1):
            for j in range(i + 1, len(self.bodies)):
                d = self.bodies[i].distance_to(self.bodies[j])
                if d < closest_dist:
                    closest_dist = d
        return closest_dist


class Simulation:
    def __init__(self, init_filepath: str, energy_plot_filename: str = None) -> None:
        self.system: System = None
        self.dt: float = None
        self.t: float = 0
        self.T: float = None

        self.energy_plot_filename = energy_plot_filename

        self.window_surf: pygame.Surface = None
        self.background_surf: pygame.Surface = None
        self.draw_surf: pygame.Surface = None
        self.my_font = None
        self.slider_scale = None
        self.slider_screen_refresh_rate = None
        self.reset_button = None
        self.clock = None

        self.text_refresh_rate: int = None
        self.init_filepath: str = init_filepath

        self.init_pygame()
        self.setup_system_from_file()

    def init_pygame(self) -> None:
        pygame.init()

        self.window_surf = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.background_surf.fill(pygame.Color("#000000"))

        self.draw_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.draw_surf.fill(pygame.Color("#000000"))

        pygame.display.set_caption("n-Body Simulation")
        pygame.font.init()
        self.my_font = pygame.font.SysFont("Arial", 15)

        self.slider_scale = Slider(
            self.draw_surf, 600, 10, 150, 10, min=1, max=250, step=1
        )

        self.slider_screen_refresh_rate = Slider(
            self.draw_surf, 800, 10, 150, 10, min=1, max=1000, step=5
        )

        self.slider_screen_refresh_rate.setValue(250)

        self.reset_button = Button(
            self.draw_surf,
            400,
            10,
            50,
            50,
            text="Reset",
            fontSize=15,
            margin=5,
            inactiveColour=(255, 0, 0),
            pressedColour=(0, 255, 0),
            radius=5,
            onClick=lambda: self.reset_simulation(),
        )

        self.clock = pygame.time.Clock()

    def setup_system_from_file(self) -> None:
        bodies = []

        with open(self.init_filepath, "r") as f:
            lines = f.readlines()
            dt, T, scale, screen_refresh_rate = list(map(float, lines[0].split(",")))
            self.slider_scale.setValue(scale)
            self.slider_screen_refresh_rate.setValue(screen_refresh_rate)

            if T == -1:
                T = math.inf

            for line in lines[1:]:
                m, x, y, vx, vy = list(map(float, line.split(",")))

                bodies.append(Body(m, x, y, vx, vy))

        self.dt = dt
        self.T = T

        self.text_refresh_rate = round(
            (1 / 15) / self.dt
        )  # updates text 20 times per second
        self.system = System(bodies, dt)

    def reset_simulation(self) -> None:
        self.t = 0
        self.init_pygame()
        self.setup_system_from_file()

    def transform_coords(self, x: float, y: float) -> tuple[float, float]:
        scale = self.slider_scale.getValue()

        tx = scale * x + self.draw_surf.get_width() / 2
        ty = self.draw_surf.get_height() - (scale * y + self.draw_surf.get_height() / 2)

        return tx, ty

    def display_system(self) -> None:
        self.draw_surf.fill(
            (0, 0, 0),
            rect=pygame.Rect(
                0, 100, self.draw_surf.get_width(), self.draw_surf.get_height() - 100
            ),
        )

        for body in self.system.bodies:
            tx, ty = self.transform_coords(body.x, body.y)

            if 0 <= ty <= 100:
                continue

            pygame.draw.circle(self.draw_surf, body.color, (tx, ty), body.r)
            self.display_indiv_text(
                f"Speed: {round(body.speed, 3)}", tx - 10, ty + body.r + 20
            )

            for i, (prev_x, prev_y) in enumerate(body.hist):
                pygame_color = pygame.Color(body.color)
                pygame_color.a = round(255 * (0.95**i))

                prev_tx, prev_ty = self.transform_coords(prev_x, prev_y)

                if math.sqrt((tx - prev_tx) ** 2 + (ty - prev_ty) ** 2) > body.r + 1:
                    pygame.draw.circle(
                        self.draw_surf,
                        pygame_color,
                        (prev_tx, prev_ty),
                        max(round(body.r / 2 * (0.9**i)), 1),
                    )

    def display_indiv_text(self, s: str, x: int, y: int) -> None:
        font_surf = self.my_font.render(s, False, (255, 255, 255))
        self.draw_surf.blit(font_surf, (x, y))

    def display_text(self, dt: float) -> None:
        self.draw_surf.fill(
            (0, 0, 0), rect=pygame.Rect(0, 0, self.draw_surf.get_width(), 100)
        )

        self.display_indiv_text(f"FPS: {round(min(self.clock.get_fps(), 1000))}", 10, 5)
        self.display_indiv_text(f"Potential Energy: {round(self.system.Us[-1], 3)}", 10, 30)
        self.display_indiv_text(f"Kinetic Energy: {round(self.system.Ks[-1], 3)}", 10, 55)
        self.display_indiv_text(f"Total Energy: {round(self.system.Es[-1], 3)}", 10, 80)
        self.display_indiv_text(f"Time: {round(self.t, 3)} s", 200, 10)
        self.display_indiv_text(f"dt: {round(dt / 1e-3, 4)} ms", 200, 30)
        self.display_indiv_text(
            f"Simulation Scale: {self.slider_scale.getValue()}", 600, 25
        )
        self.display_indiv_text(
            f"Screen Refresh Rate: {self.slider_screen_refresh_rate.getValue()}",
            800,
            25,
        )

    def run_simulation(self) -> None:
        running = True
        iter = 0
        frames_saved = 0
        while running:
            if self.t > self.T:
                running = False

            # TODO: add a collision check!

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

            self.reset_button.listen(events)
            # update bodies

            dt = self.system.update()
            self.t += dt

            # draw bodies and display energy on pygame screen
            screen_update = False

            '''
            Uncomment to create GIF.

            if self.t > (frames_saved * (1/30)) and frames_saved < 180:
                pygame.image.save(self.window_surf, f"figure_8_gif/frames_{frames_saved}.png")
                frames_saved += 1
            '''

            if iter % self.slider_screen_refresh_rate.getValue() == 0:
                self.display_system()
                screen_update = True

            if iter % self.text_refresh_rate == 0:
                self.display_text(dt)
                self.reset_button.draw()
                screen_update = True

            if screen_update:
                self.window_surf.blit(self.background_surf, (0, 0))
                self.window_surf.blit(self.draw_surf, (0, 0))

                pw.update(events)
                pygame.display.update()

                self.clock.tick()

            iter += 1
        pygame.quit()
        self.system.plot_energies(energy_plot_filename=self.energy_plot_filename)


if __name__ == "__main__":
    sim_init_filename = sys.argv[1]
    energy_plot_filename = None

    if len(sys.argv) > 2:
        energy_plot_filename = sys.argv[2]

    sim = Simulation(sim_init_filename, energy_plot_filename=energy_plot_filename)
    sim.run_simulation()
