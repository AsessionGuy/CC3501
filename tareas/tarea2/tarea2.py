import os
from pathlib import Path

import networkx as nx
import numpy as np
import pyglet.window
import trimesh as tm
from OpenGL import GL
from networkx import DiGraph

import grafica.transformations as tr
import shapes

WIDTH = 800
HEIGHT = 600


class Controller(pyglet.window.Window):
    def __init__(self, title):
        super().__init__()
        self.set_minimum_size(280, 240)
        self.set_caption(title)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.program_state = {"total_time": 0.0}

        GL.glClearColor(1, 1, 1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.key_handler[key]


class Model:
    def __init__(self, position_data, index_data=None):
        self.position_data = position_data
        self.index_data = index_data
        self.pipeline = None
        self.gpu_data = None

        if index_data is not None:
            self.index_data = np.array(index_data, dtype=np.uint32)

    def init_gpu_data(self, pipeline):
        self.pipeline = pipeline

        if self.index_data is not None:
            self.gpu_data = self.pipeline.vertex_list_indexed(len(self.position_data) // 3, GL.GL_TRIANGLES,
                                                              self.index_data)
        else:
            self.gpu_data = self.pipeline.vertex_list(len(self.position_data) // 3, GL.GL_TRIANGLES)

        self.gpu_data.position[:] = self.position_data

    def draw(self, mode=GL.GL_TRIANGLES):
        self.gpu_data.draw(mode)


class Mesh(Model):
    def __init__(self, asset_path):
        mesh_data = tm.load(asset_path)
        mesh_scale = tr.uniformScale(2.0 / mesh_data.scale)
        mesh_translate = tr.translate(*-mesh_data.centroid)
        mesh_data.apply_transform(mesh_scale @ mesh_translate)

        vertex_data = tm.rendering.mesh_to_vertexlist(mesh_data)
        indices = vertex_data[3]
        positions = vertex_data[4][1]

        super().__init__(positions, indices)


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


class SceneGraph:
    graph: DiGraph

    def __init__(self, cam=None):
        self.graph = nx.DiGraph(root="root")
        self.add_node("root")
        self.camera = cam

    def add_node(self,
                 name,
                 attach_to=None,
                 mesh=None,
                 color=None,
                 transform=tr.identity(),
                 position=None,
                 rotation=None,
                 scale=None,
                 mode=GL.GL_TRIANGLES):
        if scale is None:
            scale = [1, 1, 1]
        if rotation is None:
            rotation = [0, 0, 0]
        if position is None:
            position = [0, 0, 0]
        if color is None:
            color = [1, 1, 1]
        self.graph.add_node(
            name,
            mesh=mesh,
            color=color,
            transform=transform,
            position=np.array(position, dtype=np.float32),
            rotation=np.array(rotation, dtype=np.float32),
            scale=np.array(scale, dtype=np.float32),
            mode=mode)
        if attach_to is None:
            attach_to = "root"

        self.graph.add_edge(attach_to, name)

    def __getitem__(self, name):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")

        return self.graph.nodes[name]

    def __setitem__(self, name, value):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")

        self.graph.nodes[name] = value

    def get_transform(self, node):
        node = self.graph.nodes[node]
        transform = node["transform"]
        translation_matrix = tr.translate(node["position"][0], node["position"][1], node["position"][2])
        rotation_matrix = tr.rotationX(node["rotation"][0]) @ tr.rotationY(node["rotation"][1]) @ tr.rotationZ(
            node["rotation"][2])
        scale_matrix = tr.scale(node["scale"][0], node["scale"][1], node["scale"][2])
        return transform @ translation_matrix @ rotation_matrix @ scale_matrix

    def draw(self):
        root_key = self.graph.graph["root"]
        edges = list(nx.edge_dfs(self.graph, source=root_key))
        transformations = {root_key: self.get_transform(root_key)}

        for src, dst in edges:
            current_node = self.graph.nodes[dst]

            if dst not in transformations:
                transformations[dst] = transformations[src] @ self.get_transform(dst)

            if current_node["mesh"] is not None:
                current_pipeline = current_node["mesh"].pipeline
                current_pipeline.use()

                if self.camera is not None:
                    if "u_view" in current_pipeline.uniforms:
                        current_pipeline["u_view"] = self.camera.get_view()

                    if "u_projection" in current_pipeline.uniforms:
                        current_pipeline["u_projection"] = self.camera.get_projection()

                current_pipeline["u_model"] = np.reshape(transformations[dst], (16, 1), order="F")

                if "u_color" in current_pipeline.uniforms:
                    current_pipeline["u_color"] = np.array(current_node["color"], dtype=np.float32)
                current_node["mesh"].draw(current_node["mode"])


class Car:
    def __init__(self, cam, transform=None, color=shapes.RED):
        if transform is None:
            transform = [[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]
        else:
            transform = np.array([[1, -1.75, 0], [1, 1, 1], [0, 0, -np.pi / 9]]) + transform
        self.graph = SceneGraph(cam)

        self.graph.add_node("car",
                            position=transform[0],
                            scale=transform[1],
                            rotation=transform[2])

        self.graph.add_node("chassis",
                            attach_to="car",
                            mesh=chassis_mesh,
                            color=color,
                            scale=[3, 3, 3],
                            position=[0, -0.25, 0])

        self.graph.add_node("wheels", attach_to="car", scale=[1, 1, 1], position=[0, -1, 0])

        self.graph.add_node("forward_wheels", attach_to="wheels", position=[1.75, 0, 0])
        self.graph.add_node("wheel0",
                            attach_to="forward_wheels",
                            mesh=wheel_mesh,
                            color=shapes.GRAY,
                            position=[0, 0, 1.25])
        self.graph.add_node("wheel1",
                            attach_to="forward_wheels",
                            mesh=wheel_mesh,
                            color=shapes.GRAY,
                            position=[0, 0, -1.25])

        self.graph.add_node("backward_wheels", attach_to="wheels", position=[-1.75, 0, 0])
        self.graph.add_node("wheel2",
                            attach_to="backward_wheels",
                            mesh=wheel_mesh,
                            color=shapes.GRAY,
                            position=[0, 0, 1.25])
        self.graph.add_node("wheel3",
                            attach_to="backward_wheels",
                            mesh=wheel_mesh,
                            color=shapes.GRAY,
                            position=[0, 0, -1.2])

        self.graph.add_node("platform", attach_to="root")
        self.graph.add_node("platform_mesh",
                            attach_to="platform",
                            mesh=platform_mesh,
                            color=shapes.BLACK,
                            scale=[4, 6, 4],
                            rotation=[0, np.pi / 2, 0],
                            position=[0, -4, 0])

    def draw(self):
        self.graph.draw()


if __name__ == "__main__":
    controller = Controller("Tarea 1 - Andres Gallardo Cornejo")

    with open(Path(os.path.dirname(__file__)) / "./color_mesh.vert") as f:
        vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "./color_mesh.frag") as f:
        fragment_source_code = f.read()

    mesh_pipeline = pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vertex_source_code, "vertex"),
        pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    )
    mesh_pipeline.use()

    chassis_mesh = Mesh("./chassis.obj")
    chassis_mesh.init_gpu_data(mesh_pipeline)
    wheel_mesh = Mesh("./wheel.obj")
    wheel_mesh.init_gpu_data(mesh_pipeline)
    garage_mesh = Mesh("./garage.obj")
    garage_mesh.init_gpu_data(mesh_pipeline)
    platform_mesh = Mesh("./platform.obj")
    platform_mesh.init_gpu_data(mesh_pipeline)

    camera = OrbitCamera(20)
    car1 = Car(cam=camera)
    car2 = Car(cam=camera, transform=np.array([[0, 0, -5.5], [-1, -1, -1], [0, 0, 0]]), color=shapes.YELLOW)
    car3 = Car(cam=camera, transform=np.array([[0, 0, 5.5], [-1, -1, -1`], [0, 0, 0]]), color=shapes.GREEN)

    environment = SceneGraph(cam=camera)
    environment.add_node("environment")

    environment.add_node("garage", attach_to="environment")
    environment.add_node("garage_mesh",
                         attach_to="garage",
                         mesh=garage_mesh,
                         color=shapes.BLACK,
                         scale=[12, 18, 12],
                         position=[0, 0.325, 0])

    selected_car = car1


    def update(dt):
        controller.program_state["total_time"] += dt

        # selected_car.graph.graph.nodes["car"]["rotation"] =
        # selected_car.graph.graph.nodes["car"]["position"] =

        if controller.is_key_pressed(pyglet.window.key.A):
            camera.phi -= dt * 5
        if controller.is_key_pressed(pyglet.window.key.D):
            camera.phi += dt * 5
        if controller.is_key_pressed(pyglet.window.key.S):
            camera.distance += dt * 10
        if controller.is_key_pressed(pyglet.window.key.W):
            camera.distance -= dt * 10

        camera.update()


    @controller.event
    def on_draw():
        controller.clear()
        car1.draw()
        car2.draw()
        car3.draw()
        environment.draw()


    pyglet.clock.schedule_interval(update, 1 / 60)
    pyglet.app.run()
