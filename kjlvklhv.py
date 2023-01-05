import pygame,sys,pymunk
import math
pygame.init()

AU = 149.6*10**9

class space_object():
    size=0
    mass=0
    moment=0
    pos_x=0
    pos_y=0
    SCALE = 250 / AU  # 1AU = 100 pixels
    def __init__(self,space,screen,mass,size,pos_x,pos_y,moment):
        self.mass=mass
        self.moment=moment
        self.size=size
        self.pos_x=pos_x
        self.pos_y=pos_y
        self.screen = screen
        self.body=pymunk.Body(0,0,pymunk.Body.STATIC)
        self.body.position=(pos_x,pos_y)
        self.space=space
    def get_mass(self):
        return self.body.mass
    def draw(self):
        c_x=self.screen.get_width()/2
        c_y=self.screen.get_height()/2
        x=c_x-self.body.position.x * self.SCALE
        y=c_y-self.body.position.y * self.SCALE
#        print(self.body.position)
        pygame.draw.circle(self.screen,(0,0,0),(x,y),self.size)

class Sun(space_object):
    def __init__(self,space,screen,mass,pos_x,pos_y,moment):
        super().__init__(space,screen,mass,25,pos_x,pos_y,moment)
        self.body=pymunk.Body(self.mass,self.moment,body_type = pymunk.Body.STATIC)
        #self.body.mass=self.mass
        self.body.position=(pos_x,pos_y)
        self.shape=pymunk.Circle(self.body,self.size)
        self.space.add(self.body,self.shape)
    def get_mass(self):
        
        return self.mass
class Body(space_object):
    def __init__(self,space,screen,mass,pos_x,pos_y,moment):
        super().__init__(space,screen,mass,5,pos_x,pos_y,moment)
        self.body=pymunk.Body(self.mass,self.moment,body_type = pymunk.Body.DYNAMIC)
        self.body.velocity=( -10 * 1000,29.783 * 1000)
        self.body.mass=self.mass
        self.body.position=(pos_x,pos_y)
        self.shape=pymunk.Circle(self.body,self.size)
        self.space.add(self.body,self.shape)


    


def draw_objects(objects):
    for object in objects:
        object.draw()

        
def gravity(m1,m2,xs,xb,ys,yb,):
    G = 6.67428*10**(-11)
    d_x=(xb-xs)
    d_y=(yb-ys)
    o=math.atan2(abs(d_y),abs(d_x))
    
    r=math.sqrt((d_x**2)+((d_y**2)))
    v=1
    w=1
    if d_y>0:
        w=1
    else:
        w=-1
    if d_x<0:
        v=-1
    else:
        v=1
    gravit_x=-G*((m1*m2)/(r**2))*math.cos(o)
    gravit_y=-G*((m1*m2)/(r**2))*math.sin(o)
    print(gravit_x,gravit_y)
    return (gravit_x,gravit_y)

screen = pygame.display.set_mode((800,600), pygame.RESIZABLE)

clock = pygame.time.Clock()
space = pymunk.Space()
objects=[]
sun = Sun(space,screen,1.98892 * 10**30,0,0,100)
body = Body(space,screen,5.9742 * 10**24,1*AU,0,100)
objects.append(sun)
objects.append(body)
space.gravity = gravity(sun.get_mass(),body.get_mass(),(sun.body.position.x),(body.body.position.x),(sun.body.position.y),(body.body.position.y))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((217,217,217)) 
    draw_objects(objects)
    space.gravity = gravity(sun.get_mass(),body.get_mass(),(sun.body.position.x),(body.body.position.x),(sun.body.position.y),(body.body.position.y))
    space.step(3600*24)
    pygame.display.update()
    clock.tick(120)
