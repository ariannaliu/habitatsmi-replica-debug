import argparse
from settings import default_sim_settings, make_cfg

import math
import os
import random
import time
import numpy as np
from PIL import Image
import habitat_sim
import habitat_sim.agent

# python3 ../examples_debug/navigation.py --scene /home/zliu/3d2img/ReplicaData/hotel_0/habitat/mesh_semantic.ply --save_png --depth_sensor --max_frames 30

class Runner:
    def __init__(self, sim_settings):
        self.set_sim_settings(sim_settings)
    
    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()
    
    def print_semantic_scene(self):
        if self._sim_settings["print_semantic_scene"]:
            scene = self._sim.semantic_scene
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
            input("Press Enter to continue...")
    
    def init_agent_state(self, agent_id):
        # Set agent start state Here
        agent = self._sim.initialize_agent(agent_id)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
        agent.set_state(agent_state)

        start_state = agent.get_state()
        
        if not self._sim_settings["silent"]:
            print(
                "start_state.position\t",
                start_state.position,
                "start_state.rotation\t",
                start_state.rotation,
            )
        return start_state

    def init_common(self):
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (
            not os.path.exists(scene_file)
            and scene_file == default_sim_settings["scene"]
        ):
            print(
                "Test scenes not downloaded locally, downloading and extracting now..."
            )
            data_downloader.main(["--uids", "habitat_test_scenes"])
            print("Downloaded and extracted test scenes data.")
        
        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = habitat_sim.Simulator(self._cfg)

        # the randomness is needed when choosing the actions
        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        return self.init_agent_state(self._sim_settings["default_agent"])
    
    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save("test.rgba.%05d.png" % total_frames)
    
    def save_semantic_observation(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img.save("test.sem.%05d.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        depth_img.save("test.depth.%05d.png" % total_frames)

    def do_time_steps(self):
        # get the rigid object manager, which provides direct
        # access to objects
        rigid_obj_mgr = self._sim.get_rigid_object_manager()

        total_sim_step_time = 0.0
        total_frames = 0
        start_time = time.time()
        action_names = list(
            self._cfg.agents[self._sim_settings["default_agent"]].action_space.keys()
        )

        # load an object and position the agent for physics testing
        if self._sim_settings["enable_physics"]:
            self.init_physics_test_scene(
                num_objects=self._sim_settings.get("num_objects")
            )
            print("active object names: " + str(rigid_obj_mgr.get_object_handles()))

        time_per_step = []

        position_log = []
        rotation_log = []

        while total_frames < self._sim_settings["max_frames"]:
            if total_frames == 1:
                start_time = time.time()
            action = random.choice(action_names)
            if not self._sim_settings["silent"]:
                print("action", action)
            
            start_step_time = time.time()

            # apply kinematic or dynamic control to all objects based on their MotionType
            if self._sim_settings["enable_physics"]:
                obj_names = rigid_obj_mgr.get_object_handles()
                for obj_name in obj_names:
                    rand_nudge = np.random.uniform(-0.05, 0.05, 3)
                    obj = rigid_obj_mgr.get_object_by_handle(obj_name)
                    if obj.motion_type == MotionType.KINEMATIC:
                        obj.translate(rand_nudge)
                    elif obj.motion_type == MotionType.DYNAMIC:
                        obj.apply_force(rand_nudge, np.zeros(3))
            
            # get "interaction" time
            total_sim_step_time += time.time() - start_step_time

            observations = self._sim.step(action)
            time_per_step.append(time.time() - start_step_time)

            # get simulation step time without sensor observations
            total_sim_step_time += self._sim._previous_step_time

            if self._sim_settings["save_png"]:
                if self._sim_settings["color_sensor"]:
                    self.save_color_observation(observations, total_frames)
                if self._sim_settings["depth_sensor"]:
                    self.save_depth_observation(observations, total_frames)
                if self._sim_settings["semantic_sensor"]:
                    self.save_semantic_observation(observations, total_frames)
            
            state = self._sim.last_state()

            if not self._sim_settings["silent"]:
                print("position\t", state.position, "\t", "rotation\t", state.rotation)
                position_log.append(state.position)
                rotation_log.append([state.rotation.w,state.rotation.x,state.rotation.y,state.rotation.z])

            if self._sim_settings["compute_shortest_path"]:
                self.compute_shortest_path(
                    state.position, self._sim_settings["goal_position"]
                )

            if self._sim_settings["compute_action_shortest_path"]:
                self._action_path = self.greedy_follower.find_path(
                    self._sim_settings["goal_position"]
                )
                print("len(action_path)", len(self._action_path))

            if (
                self._sim_settings["semantic_sensor"]
                and self._sim_settings["print_semantic_mask_stats"]
            ):
                self.output_semantic_mask_stats(observations, total_frames)

            total_frames += 1

        end_time = time.time()
        perf = {"total_time": end_time - start_time}
        perf["frame_time"] = perf["total_time"] / total_frames
        perf["fps"] = 1.0 / perf["frame_time"]
        perf["time_per_step"] = time_per_step
        perf["avg_sim_step_time"] = total_sim_step_time / total_frames

        np.savetxt("./position_log.txt", np.array(position_log))
        # print(rotation_log)
        np.savetxt("./rotation_log.txt", np.array(rotation_log))

        return perf
            

    def example(self):
        start_state = self.init_common()
        # print("AAAAAAA", self._cfg.sensor_specifications['hfov'])
        self.print_semantic_scene()
        perf = self.do_time_steps()

        self._sim.close()
        del self._sim

        return perf


        
        




def make_settings(args):
    settings = default_sim_settings.copy()
    settings["max_frames"] = args.max_frames
    settings["width"] = args.width
    settings["height"] = args.height
    settings["scene"] = args.scene
    settings["save_png"] = args.save_png
    settings["sensor_height"] = args.sensor_height
    settings["color_sensor"] = not args.disable_color_sensor
    settings["semantic_sensor"] = args.semantic_sensor
    settings["depth_sensor"] = args.depth_sensor
    settings["print_semantic_scene"] = args.print_semantic_scene
    settings["print_semantic_mask_stats"] = args.print_semantic_mask_stats
    settings["compute_shortest_path"] = args.compute_shortest_path
    settings["compute_action_shortest_path"] = args.compute_action_shortest_path
    settings["seed"] = args.seed
    settings["silent"] = args.silent
    settings["enable_physics"] = args.enable_physics
    settings["physics_config_file"] = args.physics_config_file
    settings["frustum_culling"] = not args.disable_frustum_culling
    settings["recompute_navmesh"] = args.recompute_navmesh
    return settings



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=default_sim_settings["scene"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--save_png", action="store_true") 
    parser.add_argument("--sensor_height", type=float, default=0.5)
    parser.add_argument("--disable_color_sensor", action="store_true")
    parser.add_argument("--semantic_sensor", action="store_true")
    parser.add_argument("--depth_sensor", action="store_true")
    parser.add_argument("--print_semantic_scene", action="store_true")
    parser.add_argument("--print_semantic_mask_stats", action="store_true")
    parser.add_argument("--compute_shortest_path", action="store_true")
    parser.add_argument("--compute_action_shortest_path", action="store_true")
    parser.add_argument("--recompute_navmesh", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--test_fps_regression", type=int, default=0)
    parser.add_argument("--enable_physics", action="store_true")
    parser.add_argument(
        "--physics_config_file",
        type=str,
        default=default_sim_settings["physics_config_file"],
    )
    parser.add_argument("--disable_frustum_culling", action="store_true")
    args = parser.parse_args()

    settings = make_settings(args)
    print("BBBBBBB", settings["hfov"])

    runner = Runner(settings)
    perf = runner.example()

    print(" ========================= Performance ======================== ")
    print(
        " %d x %d, total time %0.2f s,"
        % (settings["width"], settings["height"], perf["total_time"]),
        "frame time %0.3f ms (%0.1f FPS)" % (perf["frame_time"] * 1000.0, perf["fps"]),
    )
    print(" ============================================================== ")