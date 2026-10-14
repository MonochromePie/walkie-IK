import mujoco
import mujoco.viewer

def main():
    model = mujoco.MjModel.from_xml_path('mujoco/walker_scene.xml')
    data = mujoco.MjData(model)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()