import pyglet
from OpenGL import GL
import numpy as np
import os
from pathlib import Path

if __name__ == "__main__":

    win = pyglet.window.Window(800, 600)
    
    # seteamos el color de fondo
    GL.glClearColor(0.75, 0.75, 0.75, 1.0)

    # Dibujaremos una piramide de base triangular
    # Para ello necesitamos 4 triangulos equilateros, cada uno
    # con 3 vertices y su respectivo color.
    # Los vertices del poliedro deben estar ubicados en:
    # (0,0,0), (0.866, 0.5, 0.0), (0.866, -0.5, 0.0), (0.577, 0, 0.577)
    
    vertices_t1 = np.array(
        [0.866, 0.5, 0.0,  
          0.866, -0.5, 0.0,
          0.0,  0.0, 0.0,  
        ],
        dtype=np.float32,
    )

    vertex_colors_t1 = np.array(
        [61, 0, 102,  
          102, 0, 204,
          61, 0, 102,   
        ],
        dtype=np.float32,
    )
    
    vertices_t2 = np.array(
        [0.866, 0.5, 0.0,  
          0.577, 0, 0.577,
          0.866, -0.5, 0.0, 
        ],
        dtype=np.float32,
    )

    vertex_colors_t2 = np.array(
        [61, 0, 102,  
          102, 0, 204,
          61, 0, 102,   
        ],
        dtype=np.float32,
    )
    
    vertices_t3 = np.array(
        [0.866, -0.5, 0.0,  
          0.577, 0, 0.577,
          0.0, 0.0, 0.0, 
        ],
        dtype=np.float32,
    )

    vertex_colors_t3 = np.array(
        [61, 0, 102,  
          102, 0, 204,
          61, 0, 102,   
        ],
        dtype=np.float32,
    )
    
    vertices_t4 = np.array(
        [0.866, 0.5, 0.0,  
          0.577, 0, 0.577,
          0.0,  0.0, 0.0,  
        ],
        dtype=np.float32,
    )

    vertex_colors_t4 = np.array(
        [61, 0, 102,  
          102, 0, 204,
          61, 0, 102,   
        ],
        dtype=np.float32,
    )
    
    # Los empalmamos en un solo arreglo para poder graficarlos mas facilmente
    vertices = (vertices_t1, vertices_t2, vertices_t3, vertices_t4)
    colores = (vertex_colors_t1, vertex_colors_t2, vertex_colors_t3, vertex_colors_t4)
    
    # Ahora cargamos los shaders
    with open(Path(os.path.dirname(__file__)) / "vertex_program.glsl") as f:
        vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "fragment_program.glsl") as f:
        fragment_source_code = f.read()
        
    # Y los inicializamos en la GPU
    vert_shader = pyglet.graphics.shader.Shader(vertex_source_code, "vertex")
    frag_shader = pyglet.graphics.shader.Shader(fragment_source_code, "fragment")
    
    # Creamos el pipeline, lo activamos y limpiamos la pantalla para
    # empezar a graficar
    pipeline = pyglet.graphics.shader.ShaderProgram(vert_shader, frag_shader)
    pipeline.use()
    win.clear()
    
    # Ahora le entregaremos los datos de nuestro objeto a la GPU
    # recorriendo el arreglo de vertices y colores y graficandolos
    for i in range(4):
    
        gpu_data = pipeline.vertex_list(3, GL.GL_TRIANGLES)
        gpu_data.position[:] = vertices[i]/4
        gpu_data.color[:] = colores[i]/255
        gpu_data.draw(GL.GL_TRIANGLES)
        
    pyglet.app.run()
