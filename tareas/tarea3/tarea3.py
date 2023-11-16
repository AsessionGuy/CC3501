import numpy as np
import pyglet.window
from Box2D import b2World
from OpenGL import GL

import grafica.transformations as tr
import helpers
from drawables import Material, SpotLight, Texture, DirectionalLight
from scene_graph import SceneGraph
from shapes import Square

WIDTH = 1280
HEIGHT = 720


def update_scene(dt):
    controller.program_state["total_time"] += dt

    if keyboard[pyglet.window.key.A]:
        camera.phi -= dt * 5
    if keyboard[pyglet.window.key.D]:
        camera.phi += dt * 5
    if keyboard[pyglet.window.key.S]:
        camera.distance += dt * 10
    if keyboard[pyglet.window.key.W]:
        if camera.distance > 5:
            camera.distance -= dt * 10

    if controller.program_state["pre_selected"]:
        car_list.current_car.graph.graph.nodes[car_list.current_car.name]["rotation"] += np.array([0, dt * 7.0, 0])

    car_list.current_car.graph.graph.nodes[car_list.current_car.name]["scale"] += dt * 3.0

    camera.update()
    car_list.current_car.update()


def track_update(dt):
    pass


def draw_scene():
    selection_cars.draw()
    selection_environment.draw()


class CarItem:
    def __init__(self, car):
        self.car = car
        self.next = None
        self.prev = None


class CarList:
    def __init__(self, first_car):
        self._cars = [first_car]
        self.first_car = first_car
        self.current_car = first_car
        self._unselected_cars = []

    def add_car(self, car):
        car.prev = self._cars[-1]
        self._cars[-1].next = car
        self._cars.append(car)
        self._cars[-1].next = self.first_car
        self._unselected_cars.append(car)
        self.first_car.prev = self._cars[-1]

    def next_car(self):
        self._unselected_cars.append(self.current_car)
        self.current_car = self.current_car.next
        self._unselected_cars.remove(self.current_car)

    def prev_car(self):
        self._unselected_cars.append(self.current_car)
        self.current_car = self.current_car.prev
        self._unselected_cars.remove(self.current_car)

    def get_unselected_cars(self):
        for car in self._unselected_cars:
            yield car


class Controller(pyglet.window.Window):
    def __init__(self, title):
        super().__init__()
        self.set_minimum_size(280, 240)
        self.set_caption(title)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.program_state = {"total_time": 0.0, "camera": None, "scene": None, "pre_selected": False,
                              "selected": False, "vel_iters": 6, "pos_iters": 2, "car_body": None, "selected_car": None,
                              "forwards": True}

        GL.glClearColor(0, 0, 0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)


class Model:
    def __init__(self, position_data, uv_data=None, normal_data=None, index_data=None):
        self.position_data = position_data
        self.uv_data = uv_data
        self.normal_data = normal_data

        self.index_data = index_data
        if index_data is not None:
            self.index_data = np.array(index_data, dtype=np.uint32)

        self.gpu_data = None

    def init_gpu_data(self, pipeline):

        size = len(self.position_data)
        count = 3

        if "texCoord" in pipeline.attributes:
            size += len(self.uv_data)
            count += 2

        if "normal" in pipeline.attributes:
            size += len(self.normal_data)
            count += 3

        if self.index_data is not None:
            self.gpu_data = pipeline.vertex_list_indexed(size // count, GL.GL_TRIANGLES, self.index_data)
        else:
            self.gpu_data = pipeline.vertex_list(size // count, GL.GL_TRIANGLES)

        self.gpu_data.position[:] = self.position_data
        if "texCoord" in pipeline.attributes:
            self.gpu_data.texCoord[:] = self.uv_data

        if "normal" in pipeline.attributes:
            self.gpu_data.normal[:] = self.normal_data

    def draw(self, mode=GL.GL_TRIANGLES):

        self.gpu_data.draw(mode)


class Camera:
    def __init__(self, camera_type="perspective", width=WIDTH, height=HEIGHT):
        self.position = np.array([1, 0, 0], dtype=np.float32)
        self.focus = np.array([0, 0, 0], dtype=np.float32)
        self.type = camera_type
        self.width = width
        self.height = height

    def update(self):
        pass

    def get_view(self):
        lookAt_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(lookAt_matrix, (16, 1), order="F")

    def get_projection(self):
        if self.type == "perspective":
            perspective_matrix = tr.perspective(90, self.width / self.height, 0.01, 100)
        elif self.type == "orthographic":
            depth = self.position - self.focus
            depth = np.linalg.norm(depth)
            perspective_matrix = tr.ortho(-(self.width / self.height) * depth, (self.width / self.height) * depth,
                                          -1 * depth, 1 * depth, 0.01, 100)
        return np.reshape(perspective_matrix, (16, 1), order="F")

    def resize(self, width, height):
        self.width = width
        self.height = height


class OrbitCamera(Camera):
    def __init__(self, distance):
        super().__init__()
        self.distance = distance
        self.phi = np.pi / 2
        self.theta = np.pi / 2
        self.update()

    def update(self):
        if self.theta > np.pi:
            self.theta = np.pi
        elif self.theta < 0:
            self.theta = 0.0001

        self.position[0] = self.distance * np.sin(self.theta) * np.sin(self.phi)
        self.position[1] = self.distance * np.cos(self.theta)
        self.position[2] = self.distance * np.sin(self.theta) * np.cos(self.phi)


class FreeCamera(Camera):
    def __init__(self, position=[0, 0, 0], camera_type="perspective"):
        super().__init__(camera_type)
        self.position = np.array(position, dtype=np.float32)
        self.pitch = 0
        self.yaw = 0
        self.forward = np.array([0, 0, -1], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.update()

    def update(self):
        self.forward[0] = np.cos(self.yaw) * np.cos(self.pitch)
        self.forward[1] = np.sin(self.pitch)
        self.forward[2] = np.sin(self.yaw) * np.cos(self.pitch)
        self.forward = self.forward / np.linalg.norm(self.forward)

        self.right = np.cross(self.forward, np.array([0, 1, 0], dtype=np.float32))
        self.right = self.right / np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

        self.focus = self.position + self.forward


class Car:
    def __init__(self, controller, transform=None, material=Material(), attach_to=None, name="car"):
        self.name = name
        self.controller = controller
        self.material = material
        if transform is None:
            transform = [[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]
        else:
            transform = np.array([[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]) + transform
        self.default_transform = transform
        if attach_to is None:
            self.graph = SceneGraph(controller=controller)
            self.graph.add_node(name,
                                position=transform[0],
                                scale=transform[1],
                                rotation=transform[2])
        else:
            attach_to.add_node(name,
                               attach_to="root",
                               position=transform[0],
                               scale=transform[1],
                               rotation=transform[2])
            self.graph = attach_to

        self.graph.add_node(name + "chassis",
                            attach_to=name,
                            mesh=chassis_mesh,
                            scale=[3, 3, 3],
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, -0.25, 0],
                            material=material)

        self.graph.add_node(name + "wheels", attach_to=name, scale=[1, 1, 1], position=[0, -1, 0])

        self.graph.add_node(name + "forward_wheels", attach_to=name + "wheels", position=[1.75, 0, 0])
        self.graph.add_node(name + "wheel0",
                            attach_to=name + "forward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, 1.25],
                            material=mat_black_rubber)
        self.graph.add_node(name + "wheel1",
                            attach_to=name + "forward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, -1.25],
                            material=mat_black_rubber)

        self.graph.add_node(name + "backward_wheels", attach_to=name + "wheels", position=[-1.75, 0, 0])
        self.graph.add_node(name + "wheel2",
                            attach_to=name + "backward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, 1.25],
                            material=mat_black_rubber)
        self.graph.add_node(name + "wheel3",
                            attach_to=name + "backward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, -1.35],
                            material=mat_black_rubber)

    def draw(self):
        self.graph.draw()

    def update(self):
        if (self.graph.graph.nodes[self.name]["scale"] > 1.0).any():
            self.graph.graph.nodes[self.name]["scale"] = np.array([1.0, 1.0, 1.0])
        # if np.pi / 4 < self.graph.graph.nodes[self.name + "forward_wheels"]["rotation"][1]:
        #     self.graph.graph.nodes[self.name + "forward_wheels"]["rotation"][1] = np.pi / 4
        # elif self.graph.graph.nodes[self.name + "forward_wheels"]["rotation"][1] < - np.pi / 4:
        #     self.graph.graph.nodes[self.name + "forward_wheels"]["rotation"][1] = - np.pi / 4
        # if np.pi / 4 < self.graph.graph.nodes[self.name + "wheel1"]["rotation"][1]:
        #     self.graph.graph.nodes[self.name + "wheel1"]["rotation"][1] = np.pi / 4
        # elif self.graph.graph.nodes[self.name + "wheel1"]["rotation"][1] < - np.pi / 4:
        #     self.graph.graph.nodes[self.name + "wheel1"]["rotation"][1] = - np.pi / 4

    def copy_and_attach(self, controller, attach_to, name=None):
        if name is None:
            name = self.name + "copy"
        return Car(controller=controller, material=self.material, attach_to=attach_to, name=name)


if __name__ == "__main__":

    camera = OrbitCamera(10)

    world = b2World(gravity=(0, 0))
    car_body = None

    controller = Controller("Tarea 3 - Andres Gallardo Cornejo")
    controller.program_state["camera"] = camera

    textured_mesh_lit_pipeline = helpers.init_pipeline(
        helpers.get_path("./textured_mesh_lit.vert"),
        helpers.get_path("./textured_mesh_lit.frag"))

    chassis_mesh = helpers.mesh_from_file("./chassis.obj")[0]["mesh"]
    wheel_mesh = helpers.mesh_from_file("./wheel.obj")[0]["mesh"]
    garage_mesh = helpers.mesh_from_file("./garage.obj")[0]["mesh"]
    platform_mesh = helpers.mesh_from_file("./platform.obj")[0]["mesh"]

    selection_cars = SceneGraph(controller=controller)

    mat_black_rubber = Material(
        ambient=[0.02, 0.02, 0.02],
        diffuse=[0.01, 0.01, 0.01],
        specular=[0.1, 0.1, 0.1],
        shininess=0.078125 * 32)

    mat_car_1 = Material(
        ambient=[0.1745, 0.01175, 0.01175],
        diffuse=[0.61424, 0.04136, 0.04136],
        specular=[0.727811, 0.626959, 0.626959],
        shininess=0.6 * 32)
    car1 = Car(controller=controller, material=mat_car_1, attach_to=selection_cars, name="car1")

    mat_car_2 = Material(
        ambient=[0.329412, 0.223529, 0.027451],
        diffuse=[0.780392, 0.568627, 0.113725],
        specular=[0.992157, 0.941176, 0.807843],
        shininess=0.21794872 * 32)
    car2 = Car(controller=controller, transform=np.array([[0, 0, -5.5], [-1, -1, -1], [0, 0, 0]]),
               material=mat_car_2, attach_to=selection_cars, name="car2")

    mat_car_3 = Material(
        ambient=[0.0215, 0.1745, 0.0215],
        diffuse=[0.07568, 0.61424, 0.07568],
        specular=[0.633, 0.727811, 0.633],
        shininess=0.6 * 32)
    car3 = Car(controller=controller, transform=np.array([[0, 0, 5.5], [-1, -1, -1], [0, 0, 0]]),
               material=mat_car_3, attach_to=selection_cars, name="car3")

    mat_car_4 = Material(
        ambient=[0.0, 0.1, 0.06],
        diffuse=[0.0, 0.50980392, 0.50980392],
        specular=[0.50196078, 0.50196078, 0.50196078],
        shininess=0.25 * 32)
    car4 = Car(controller=controller, transform=np.array([[0, 0, 5.5], [-1, -1, -1], [0, 0, 0]]),
               material=mat_car_4, attach_to=selection_cars, name="car4")

    car_list = CarList(car1)
    car_list.add_car(car2)
    car_list.add_car(car3)
    car_list.add_car(car4)

    selected_car = None

    selected_car_chassis_body = world.CreateDynamicBody(position=(1.5, 0))
    selected_car_chassis_body.CreatePolygonFixture(box=(0.5, 0.5), density=1, friction=100000000000)
    selected_car_chassis_body.linearDamping = 1.0
    selected_car_chassis_body.angularDamping = 1.0
    controller.program_state["car_body"] = selected_car_chassis_body

    selection_environment = SceneGraph(controller=controller)
    track_environment = SceneGraph(controller=controller)

    mat_environment = Material(
        ambient=[0.5, 0.5, 0.5],
        diffuse=[0.55, 0.55, 0.55],
        specular=[0.70, 0.70, 0.70],
        shininess=0.15 * 32)

    selection_environment.add_node("garage", attach_to="root")
    selection_environment.add_node("garage_mesh",
                                   attach_to="garage",
                                   mesh=garage_mesh,
                                   texture=Texture(),
                                   scale=[12, 18, 12],
                                   pipeline=textured_mesh_lit_pipeline,
                                   position=[0, 0.325, 0],
                                   material=mat_environment)

    selection_environment.add_node("lights", attach_to="root")

    selection_environment.add_node("light_directional_0",
                                   attach_to="lights",
                                   pipeline=textured_mesh_lit_pipeline,
                                   position=[-6.5, -5, 6.85],
                                   rotation=[0.75, -0.75, 0],
                                   light=SpotLight(
                                       diffuse=[1, 0, 0],
                                       specular=[0, 0, 1],
                                       ambient=[1, 1, 1],
                                       cutOff=0.80,  # siempre mayor a outerCutOff
                                       outerCutOff=0.19
                                   ))
    selection_environment.add_node("light_directional_1",
                                   attach_to="lights",
                                   pipeline=textured_mesh_lit_pipeline,
                                   position=[-6.5, -5, -6.85],
                                   rotation=[0.85, -3 * np.pi / 4, 0.75],
                                   light=SpotLight(
                                       diffuse=[1, 0, 1],
                                       specular=[0, 1, 0],
                                       ambient=[1, 1, 1],
                                       cutOff=0.80,  # siempre mayor a outerCutOff
                                       outerCutOff=0.19
                                   ))

    selection_environment.add_node("light_directional_2",
                                   attach_to="lights",
                                   pipeline=textured_mesh_lit_pipeline,
                                   position=[6.5, 5, -6.85],
                                   rotation=[-.65, np.pi - 0.75, 0],
                                   light=SpotLight(
                                       diffuse=[0.55, 0.2, 0.6],
                                       specular=[0.6, 0.5, 0.8],
                                       ambient=[0.3, 0.3, 0.3],
                                       cutOff=0.90,  # siempre mayor a outerCutOff
                                       outerCutOff=0.19
                                   ))

    selection_environment.add_node("light_directional_3",
                                   attach_to="lights",
                                   pipeline=textured_mesh_lit_pipeline,
                                   position=[6.5, 5, 6.85],
                                   rotation=[-.65, .75, 0],
                                   light=SpotLight(
                                       diffuse=[0.55, 0.2, 0.6],
                                       specular=[0.6, 0.5, 0.8],
                                       ambient=[0.3, 0.3, 0.3],
                                       cutOff=0.90,  # siempre mayor a outerCutOff
                                       outerCutOff=0.19
                                   ))

    selection_environment.add_node("platform", attach_to="root")
    selection_environment.add_node("platform_mesh",
                                   attach_to="platform",
                                   mesh=platform_mesh,
                                   texture=Texture(),
                                   scale=[4, 6, 4],
                                   pipeline=textured_mesh_lit_pipeline,
                                   rotation=[0, np.pi / 2, 0],
                                   position=[0, -4, 0],
                                   material=mat_environment)


    def update_world(dt):
        world.Step(
            dt, controller.program_state["vel_iters"], controller.program_state["pos_iters"]
        )
        world.ClearForces()


    @controller.event
    def on_draw():
        global draw_scene
        controller.clear()
        draw_scene()


    @controller.event
    def on_key_press(symbol, modifiers):

        if symbol == pyglet.window.key.E or symbol == pyglet.window.key.Q:
            if symbol == pyglet.window.key.E:
                car_list.next_car()
            elif symbol == pyglet.window.key.Q:
                car_list.prev_car()

            controller.program_state["pre_selected"] = False

            car_list.current_car.graph.graph.nodes[car_list.current_car.name]["position"] = np.array([1, -1.75, 0])
            for car in car_list.get_unselected_cars():
                car.graph.graph.nodes[car.name]["scale"] = np.array([0.0, 0.0, 0.0])
                car.graph.graph.nodes[car.name]["rotation"] = np.array([0, 0, -np.pi / 9])

        if symbol == pyglet.window.key.SPACE:
            if controller.program_state["pre_selected"]:
                controller.program_state["selected"] = True
            elif not controller.program_state["pre_selected"]:
                controller.program_state["pre_selected"] = True


    @controller.event
    def on_refresh(dt):

        global update_scene
        global draw_scene
        global on_key_press
        if controller.program_state["selected"]:
            GL.glClearColor(1, 1, 1, 1)

            selection_cars = None
            selection_environment = None

            camera = FreeCamera([0, 0, 0], "perspective")
            camera.position[1] = 8.0
            camera.pitch = -1.2
            controller.program_state["camera"] = camera

            controller.program_state["selected_car"] = car_list.current_car.copy_and_attach(controller,
                                                                                            attach_to=track_environment)

            track_environment.graph.nodes[controller.program_state["selected_car"].name]["rotation"] = np.array(
                [0, 0, 0])
            track_environment.graph.nodes[controller.program_state["selected_car"].name]["position"] = np.array(
                [1.5, 0.6, 0])
            track_environment.add_node("sun",
                                       pipeline=[textured_mesh_lit_pipeline],
                                       position=[0, 2, 0],
                                       rotation=[-np.pi / 4, 0, 0],
                                       light=DirectionalLight(diffuse=[1, 1, 1], specular=[0.25, 0.25, 0.25],
                                                              ambient=[0.15, 0.15, 0.15])
                                       )
            track_environment.add_node("floor",
                                       mesh=Model(Square["position"], Square["uv"], Square["normal"],
                                                  index_data=Square["indices"]),
                                       pipeline=textured_mesh_lit_pipeline,
                                       position=[0, -1, 0],
                                       rotation=[-np.pi / 2, 0, 0],
                                       scale=[20, 20, 20],
                                       texture=Texture(),
                                       material=Material(
                                           diffuse=[1, 1, 1],
                                           specular=[0.5, 0.5, 0.5],
                                           ambient=[0.1, 0.1, 0.1],
                                           shininess=256
                                       ))
            def draw_scene():
                track_environment.draw()

            def update_scene(dt):

                if np.linalg.norm(controller.program_state["car_body"].linearVelocity) > 1:
                    if controller.program_state["forwards"]:
                        if keyboard[pyglet.window.key.D]:
                            controller.program_state["car_body"].ApplyTorque(0.15, True)
                        if keyboard[pyglet.window.key.A]:
                            controller.program_state["car_body"].ApplyTorque(-0.15, True)

                    else:
                        if keyboard[pyglet.window.key.A]:
                            controller.program_state["car_body"].ApplyTorque(0.15, True)
                        if keyboard[pyglet.window.key.D]:
                            controller.program_state["car_body"].ApplyTorque(-0.15, True)

                    if not (keyboard[pyglet.window.key.W] or keyboard[pyglet.window.key.S]) and (
                            keyboard[pyglet.window.key.A] or keyboard[pyglet.window.key.D]):
                        if 0.10 < controller.program_state["car_body"].angularVelocity or controller.program_state[
                            "car_body"].angularVelocity < -0.10:
                            controller.program_state["car_body"].ApplyTorque(
                                0.15 * -np.sign(controller.program_state["car_body"].angularVelocity), True)

                if keyboard[pyglet.window.key.W]:
                    controller.program_state["car_body"].ApplyForceToCenter(
                        (np.cos(controller.program_state["car_body"].angle) * 5,
                         np.sin(controller.program_state["car_body"].angle) * 5), True)
                    controller.program_state["forwards"] = True


                if keyboard[pyglet.window.key.S]:
                    controller.program_state["car_body"].ApplyForceToCenter(
                        (-np.cos(controller.program_state["car_body"].angle) * 5,
                         -np.sin(controller.program_state["car_body"].angle) * 5), True)
                    controller.program_state["forwards"] = False


                print(np.linalg.norm(controller.program_state["car_body"].linearVelocity))

                if not (keyboard[pyglet.window.key.W] or keyboard[pyglet.window.key.S]):
                    if 5 < np.linalg.norm(controller.program_state["car_body"].linearVelocity):
                        controller.program_state["car_body"].ApplyForceToCenter(
                            (-np.sign(controller.program_state["car_body"].linearVelocity[0]) * np.cos(
                                controller.program_state["car_body"].angle) *
                             np.square(controller.program_state["car_body"].linearVelocity[0]),
                             -np.sign(controller.program_state["car_body"].linearVelocity[1]) * np.sin(
                                 controller.program_state["car_body"].angle) *
                             np.square(controller.program_state["car_body"].linearVelocity[1])), True)

                camera = controller.program_state["camera"]

                camera.position[0] = controller.program_state["car_body"].position[0] + np.sin(
                    controller.program_state["car_body"].angle)
                camera.position[2] = controller.program_state["car_body"].position[1] - np.cos(
                    controller.program_state["car_body"].angle)
                camera.yaw = controller.program_state["car_body"].angle

                track_environment.graph.nodes[controller.program_state["selected_car"].name][
                    "transform"] = tr.translate(
                    controller.program_state["car_body"].position[0], 0,
                    controller.program_state["car_body"].position[1]) @ tr.rotationY(
                    -controller.program_state["car_body"].angle)

                w = -np.linalg.norm(controller.program_state["car_body"].linearVelocity) / 0.5 * dt

                if - np.pi / 5 < \
                        track_environment.graph.nodes[controller.program_state["selected_car"].name + "forward_wheels"][
                            "rotation"][1] + w * controller.program_state["car_body"].angularVelocity * 3 < np.pi / 5:
                    if controller.program_state["forwards"]:
                        track_environment.graph.nodes[controller.program_state["selected_car"].name + "forward_wheels"][
                            "rotation"] = np.array(
                            [0, w * controller.program_state["car_body"].angularVelocity * 3, 0])
                    else:
                        track_environment.graph.nodes[controller.program_state["selected_car"].name + "forward_wheels"][
                            "rotation"] = np.array(
                            [0, -  w * controller.program_state["car_body"].angularVelocity * 3, 0])

                track_environment.graph.nodes[controller.program_state["selected_car"].name + "backward_wheels"][
                    "rotation"] += np.array(
                    [0, 0, controller.program_state["car_body"].angularVelocity * w])

                track_environment.graph.nodes[controller.program_state["selected_car"].name + "backward_wheels"][
                    "rotation"] += np.array(
                    [0, 0, w])

                # track_environment.graph.nodes[controller.program_state["selected_car"].name + "wheel0"][
                #     "rotation"] += np.array([controller.program_state[
                #     "car_body"].angle * dt, 0, 0])
                # track_environment.graph.nodes[controller.program_state["selected_car"].name + "wheel1"][
                #     "rotation"] += np.array([controller.program_state[
                #     "car_body"].angle * dt, 0, 0])

                camera.update()
                car_list.current_car.update()
                update_world(dt)

            controller.program_state["selected"] = False

        update_scene(dt)


    keyboard = pyglet.window.key.KeyStateHandler()
    controller.push_handlers(keyboard)
    pyglet.app.run()
