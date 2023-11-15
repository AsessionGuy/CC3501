import numpy as np
import pyglet.window
from OpenGL import GL

import grafica.transformations as tr
import helpers
from drawables import Material, SpotLight, Texture
from scene_graph import SceneGraph

WIDTH = 1280
HEIGHT = 720


class CarItem:
    def __init__(self, car):
        self.car = car
        self.next = None


class CarList:
    def __init__(self, first_car):
        self._cars = [first_car]
        self.first_car = first_car
        self.current_car = first_car
        self._unselected_cars = []

    def add_car(self, car):
        self._cars[-1].next = car
        self._cars.append(car)
        self._cars[-1].next = self.first_car
        self._unselected_cars.append(car)

    def next_car(self):
        self._unselected_cars.append(self.current_car)
        self.current_car = self.current_car.next
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
        self.program_state = {"total_time": 0.0, "is_selecting": False, "camera": None}

        GL.glClearColor(0, 0, 0, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.key_handler[key]


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
    def __init__(self):
        self.position = np.array([1, 0, 0], dtype=np.float32)
        self.focus = np.array([0, 0, 0], dtype=np.float32)
        self.width = WIDTH
        self.height = HEIGHT

    def get_view(self):
        look_at_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(look_at_matrix, (16, 1), order="F")

    def get_projection(self):
        perspective_matrix = tr.perspective(60, self.width / self.height, 0.1, 100)
        return np.reshape(perspective_matrix, (16, 1), order="F")


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


# class SceneGraph:
#     graph: DiGraph
#
#     def __init__(self, cam=None):
#         self.graph = nx.DiGraph(root="root")
#         self.add_node("root")
#         self.camera = cam
#
#     def add_node(self,
#                  name,
#                  attach_to=None,
#                  mesh=None,
#                  color=None,
#                  transform=tr.identity(),
#                  position=None,
#                  rotation=None,
#                  scale=None,
#                  mode=GL.GL_TRIANGLES):
#         if scale is None:
#             scale = [1, 1, 1]
#         if rotation is None:
#             rotation = [0, 0, 0]
#         if position is None:
#             position = [0, 0, 0]
#         if color is None:
#             color = [1, 1, 1]
#         self.graph.add_node(
#             name,
#             mesh=mesh,
#             color=color,
#             transform=transform,
#             position=np.array(position, dtype=np.float32),
#             rotation=np.array(rotation, dtype=np.float32),
#             scale=np.array(scale, dtype=np.float32),
#             mode=mode)
#         if attach_to is None:
#             attach_to = "root"
#
#         self.graph.add_edge(attach_to, name)
#
#     def __getitem__(self, name):
#         if name not in self.graph.nodes:
#             raise KeyError(f"Node {name} not in graph")
#
#         return self.graph.nodes[name]
#
#     def __setitem__(self, name, value):
#         if name not in self.graph.nodes:
#             raise KeyError(f"Node {name} not in graph")
#
#         self.graph.nodes[name] = value
#
#     def get_transform(self, node):
#         node = self.graph.nodes[node]
#         transform = node["transform"]
#         translation_matrix = tr.translate(node["position"][0], node["position"][1], node["position"][2])
#         rotation_matrix = tr.rotationX(node["rotation"][0]) @ tr.rotationY(node["rotation"][1]) @ tr.rotationZ(
#             node["rotation"][2])
#         scale_matrix = tr.scale(node["scale"][0], node["scale"][1], node["scale"][2])
#         return transform @ translation_matrix @ rotation_matrix @ scale_matrix
#
#     def draw(self):
#         root_key = self.graph.graph["root"]
#         edges = list(nx.edge_dfs(self.graph, source=root_key))
#         transformations = {root_key: self.get_transform(root_key)}
#
#         for src, dst in edges:
#             current_node = self.graph.nodes[dst]
#
#             if dst not in transformations:
#                 transformations[dst] = transformations[src] @ self.get_transform(dst)
#
#             if current_node["mesh"] is not None:
#                 current_pipeline = current_node["mesh"].pipeline
#                 current_pipeline.use()
#
#                 if self.camera is not None:
#                     if "u_view" in current_pipeline.uniforms:
#                         current_pipeline["u_view"] = self.camera.get_view()
#
#                     if "u_projection" in current_pipeline.uniforms:
#                         current_pipeline["u_projection"] = self.camera.get_projection()
#
#                 current_pipeline["u_model"] = np.reshape(transformations[dst], (16, 1), order="F")
#
#                 if "u_color" in current_pipeline.uniforms:
#                     current_pipeline["u_color"] = np.array(current_node["color"], dtype=np.float32)
#                 current_node["mesh"].draw(current_node["mode"])
#

class Car:
    def __init__(self, controller, transform=None, material=Material()):
        if transform is None:
            transform = [[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]
        else:
            transform = np.array([[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]) + transform
        self.graph = SceneGraph(controller=controller)

        self.graph.add_node("car",
                            position=transform[0],
                            scale=transform[1],
                            rotation=transform[2])

        self.graph.add_node("chassis",
                            attach_to="car",
                            mesh=chassis_mesh,
                            scale=[3, 3, 3],
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, -0.25, 0],
                            material=material)

        self.graph.add_node("wheels", attach_to="car", scale=[1, 1, 1], position=[0, -1, 0])

        self.graph.add_node("forward_wheels", attach_to="wheels", position=[1.75, 0, 0])
        self.graph.add_node("wheel0",
                            attach_to="forward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, 1.25],
                            material=mat_black_rubber)
        self.graph.add_node("wheel1",
                            attach_to="forward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, -1.25],
                            material=mat_black_rubber)

        self.graph.add_node("backward_wheels", attach_to="wheels", position=[-1.75, 0, 0])
        self.graph.add_node("wheel2",
                            attach_to="backward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, 1.25],
                            material=mat_black_rubber)
        self.graph.add_node("wheel3",
                            attach_to="backward_wheels",
                            mesh=wheel_mesh,
                            texture=Texture(),
                            pipeline=textured_mesh_lit_pipeline,
                            position=[0, 0, -1.35],
                            material=mat_black_rubber)

    def draw(self):
        self.graph.draw()

    def update(self):
        if (self.graph.graph.nodes["car"]["scale"] > 1.0).any():
            self.graph.graph.nodes["car"]["scale"] = np.array([1.0, 1.0, 1.0])


if __name__ == "__main__":

    camera = OrbitCamera(20)

    controller = Controller("Tarea 1 - Andres Gallardo Cornejo")
    controller.program_state["camera"] = camera

    textured_mesh_lit_pipeline = helpers.init_pipeline(
        helpers.get_path("./textured_mesh_lit.vert"),
        helpers.get_path("./textured_mesh_lit.frag"))

    chassis_mesh = helpers.mesh_from_file("./chassis.obj")[0]["mesh"]
    wheel_mesh = helpers.mesh_from_file("./wheel.obj")[0]["mesh"]
    garage_mesh = helpers.mesh_from_file("./garage.obj")[0]["mesh"]
    platform_mesh = helpers.mesh_from_file("./platform.obj")[0]["mesh"]

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
    car1 = Car(controller=controller, material=mat_car_1)

    mat_car_2 = Material(
        ambient=[0.329412, 0.223529, 0.027451],
        diffuse=[0.780392, 0.568627, 0.113725],
        specular=[0.992157, 0.941176, 0.807843],
        shininess=0.21794872 * 32)
    car2 = Car(controller=controller, transform=np.array([[0, 0, -5.5], [-1, -1, -1], [0, 0, 0]]),
               material=mat_car_2)

    mat_car_3 = Material(
        ambient=[0.0215, 0.1745, 0.0215],
        diffuse=[0.07568, 0.61424, 0.07568],
        specular=[0.633, 0.727811, 0.633],
        shininess=0.6 * 32)
    car3 = Car(controller=controller, transform=np.array([[0, 0, 5.5], [-1, -1, -1], [0, 0, 0]]),
               material=mat_car_3)

    car_list = CarList(car1)
    car_list.add_car(car2)
    car_list.add_car(car3)

    environment = SceneGraph(controller=controller)

    mat_environment = Material(
        ambient=[0.5, 0.5, 0.5],
        diffuse=[0.55, 0.55, 0.55],
        specular=[0.70, 0.70, 0.70],
        shininess=0.15*32)

    environment.add_node("garage", attach_to="root")
    environment.add_node("garage_mesh",
                         attach_to="garage",
                         mesh=garage_mesh,
                         texture=Texture(),
                         scale=[12, 18, 12],
                         pipeline=textured_mesh_lit_pipeline,
                         position=[0, 0.325, 0],
                         material=mat_environment)

    environment.add_node("lights", attach_to="root")

    environment.add_node("light_directional_0",
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
    environment.add_node("light_directional_1",
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

    environment.add_node("light_directional_2",
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

    environment.add_node("light_directional_3",
                         attach_to="lights",
                         pipeline=textured_mesh_lit_pipeline,
                         position=[6.5, 5, 6.85],
                         rotation=[-.65,.75,0],
                         light=SpotLight(
                             diffuse=[0.55, 0.2, 0.6],
                             specular=[0.6, 0.5, 0.8],
                             ambient=[0.3, 0.3, 0.3],
                             cutOff=0.90,  # siempre mayor a outerCutOff
                             outerCutOff=0.19
                         ))

    environment.add_node("platform", attach_to="root")
    environment.add_node("platform_mesh",
                         attach_to="platform",
                         mesh=platform_mesh,
                         texture=Texture(),
                         scale=[4, 6, 4],
                         pipeline=textured_mesh_lit_pipeline,
                         rotation=[0, np.pi / 2, 0],
                         position=[0, -4, 0],
                         material=mat_environment)

    def update(dt):
        controller.program_state["total_time"] += dt

        if controller.is_key_pressed(pyglet.window.key.A):
            camera.phi -= dt * 5
        if controller.is_key_pressed(pyglet.window.key.D):
            camera.phi += dt * 5
        if controller.is_key_pressed(pyglet.window.key.S):
            camera.distance += dt * 10
        if controller.is_key_pressed(pyglet.window.key.W):
            if camera.distance > 5:
                camera.distance -= dt * 10
        if controller.is_key_pressed(pyglet.window.key.SPACE) and not controller.program_state["is_selecting"]:
            car_list.next_car()
            car_list.current_car.graph.graph.nodes["car"]["position"] = np.array([1, -1.75, 0])
            controller.program_state["is_selecting"] = True
        elif controller.is_key_pressed(pyglet.window.key.SPACE):
            controller.program_state["is_selecting"] = True
        else:
            controller.program_state["is_selecting"] = False

        for car in car_list.get_unselected_cars():
            car.graph.graph.nodes["car"]["scale"] = np.array([0.0, 0.0, 0.0])

        car_list.current_car.graph.graph.nodes["car"]["scale"] += dt * 3.0

        camera.update()
        car_list.current_car.update()


    @controller.event
    def on_draw():
        controller.clear()
        car1.draw()
        car2.draw()
        car3.draw()
        environment.draw()

    pyglet.clock.schedule_interval(update, 1 / 60)
    pyglet.app.run()
