"""The elliptical bristlebot: its footprint, and how it moves about its centre of rotation."""
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.patches import Ellipse
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import os
import shutil

toCoR = lambda angle: np.array([-np.sin(angle), np.cos(angle)])
orient = lambda angle_rad: np.array([np.cos(angle_rad), np.sin(angle_rad)])
rotate = lambda angle_rad: np.array([[np.cos(angle_rad), -np.sin(angle_rad)],[np.sin(angle_rad), np.cos(angle_rad)]])

class EllipticalBot2D():

    id_iter = count()
    
    def __init__(self,
                r=np.array([0,0],dtype=np.float64),
                theta_rad=0.0,
                vmag=0, w=0,
                a=0.03, b=0.015,
                noise=0, npoints=101, 
                dist_rotPoint_to_CoM=np.array([0.0145, 0.0161], dtype=np.float64)):
        
        self.ID = next(self.id_iter)
  
        self.bot_width = 2*a
        self.bot_height = 2*b

        self.velocity = vmag*orient(theta_rad)
        self.ang_velocity = w
        
        self.noise = noise

        self.dist_CoMtoCoR = dist_rotPoint_to_CoM
        
        
        self.pos = r
              
        self.theta = theta_rad%(2*np.pi)

        self.npoints = npoints
        points = np.linspace(0, 2*np.pi, self.npoints)
        self.body = np.zeros([self.npoints,2], dtype=np.float64)
        self.body[:,0] = 0.5 * self.bot_width * np.cos(points)
        self.body[:,1] = 0.5 * self.bot_height * np.sin(points)
        
        self.body=self.body@rotate(theta_rad).T  
        self.body+=self.pos

        
    
    def get_CoR(self):
        """Compute and return the center of rotation (CoR) position."""
        if self.ang_velocity > 0:
            return self.pos + self.dist_CoMtoCoR[1] * toCoR(self.theta)
        elif self.ang_velocity <= 0:
            return self.pos - self.dist_CoMtoCoR[0] * toCoR(self.theta)
    
    def translate(self, dr):
        """Translate the robot by displacement vector dr."""
        self.body = self.body + dr
        self.pos = self.pos + dr
            
    def move_bot(self, dt):

        centreofRot = self.get_CoR()
        self.body = (self.body - centreofRot) @ rotate(self.ang_velocity*dt).T + centreofRot + self.velocity*dt
        self.pos = (self.pos - centreofRot) @ rotate(self.ang_velocity*dt).T + centreofRot + self.velocity*dt
        self.theta = (self.theta + self.ang_velocity*dt) % (2*np.pi)


if __name__ == "__main__":
    print("Testing EllipticalBot2D object...")
    
    init_pos = np.array([0.0, 0.0])
    init_orient = np.deg2rad(0)
    init_vmag = 0.05
    init_angVel = -5.3
    a, b = 0.06, 0.03
    noise = 0.0
    npoints = 101
    r_LC = np.array([0.0145, 0.0161], dtype=np.float64)
    TestBot = EllipticalBot2D(
        r=init_pos, 
        theta_rad=init_orient, 
        vmag=init_vmag, 
        w=init_angVel, 
        a=0.5*a, b=0.5*b,
        npoints=npoints, 
        dist_rotPoint_to_CoM=r_LC
    )

    print(f"CoM of bot: {TestBot.pos}")
    print(f"CoR of bot: {TestBot.get_CoR()}")

    print(f"Initial state with all parameters:")
    print(f"ID: {TestBot.ID}")
    print(f"Position of CoM: {TestBot.pos}")
    print(f"Orientation: {np.rad2deg(TestBot.theta)} degrees")
    print(f"CoR: {TestBot.get_CoR()}")
    print(f"Velocity: {TestBot.velocity}")
    print(f"Angular velocity: {np.rad2deg(TestBot.ang_velocity)} degrees/s")
    print(f"Bot width: {TestBot.bot_width}, Bot height: {TestBot.bot_height}")
    print(f"Noise: {TestBot.noise}")
    print(f"Number of points: {TestBot.npoints}")
    
    dt = 0.1
    steps = 250
    
    bot_data = np.zeros((steps+1, 3),dtype=np.float64)
    bot_data[0,:2] = TestBot.pos.copy()
    bot_data[0,2] = TestBot.theta

    if os.path.exists("Images"):
        shutil.rmtree("Images")
    os.makedirs("Images")

    prng = np.random.default_rng(seed=7400)
    for i in np.arange(steps):
        if i%10==0:
            fig,ax = plt.subplots(subplot_kw={"aspect":"equal"})
            ellipse = Ellipse(xy=(bot_data[i,0],bot_data[i,1]),
                            width=TestBot.bot_width, height=TestBot.bot_height,
                            angle=np.rad2deg(bot_data[i,2]), facecolor="k",
                            edgecolor="none", zorder=10)

            m = Affine2D().rotate_deg(ellipse.angle)
            ax.plot(bot_data[i,0],bot_data[i,1],color="white", marker=MarkerStyle(">","full",m), markersize=2, zorder=11)
            ax.plot(bot_data[:i-1,0], bot_data[:i-1,1], color="red", linewidth=0.5, zorder=5)
            plt.xlim(-0.5,0.5); plt.ylim(-0.5,0.5)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
        
            savefile = "Images/img_{:.2f}.png".format(i*dt)
            fig.savefig(savefile, dpi=600, format="png")
            plt.close()

        TestBot.move_bot(dt)
        bot_data[i+1,:2] = TestBot.pos.copy()
        bot_data[i+1,2] = TestBot.theta
        if i%50==0:
            print(f"Step {i}/{steps} - Position: {TestBot.pos}, Orientation: {np.rad2deg(TestBot.theta)} degrees")
    np.savetxt("bot_movement_data.txt", np.column_stack((np.arange(0,steps+1), bot_data)))