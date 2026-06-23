from dataclasses import dataclass
import time
from typing import Literal

from loguru import logger

try:
    import pyroki as pk
except Exception as e:
    print(
        f"Error importing pyroki: {e}, Please run:\n\ngit clone https://github.com/chungmin99/pyroki.git\ncd pyroki\npip install -e ."
    )
    exit()

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as onp
import viser
import viser.extras
import viser.transforms as vtf

try:
    from robot_descriptions.loaders.yourdfpy import load_robot_description
except ImportError:
    print("ImportError: robot_descriptions not found, for now:")
    print("[INFO] Will be changed to official repo once YAM and RBY1 are added and released to upstream")
    exit()


# Import the solve_ik function from pyroki_snippets
# from pyroki_snippets import solve_ik_with_multiple_targets


@dataclass
class TransformHandle:
    """Data class to store transform handles."""

    frame: viser.FrameHandle
    control: viser.TransformControlsHandle | None = None


class YAMSBaseInterface:
    """
    Base interface for dual YAM (YAMS) robot visualization.
    - This class handles two YAM robots positioned at configurable xy locations
    - Running this file allows IK control of dual virtual YAM robots in viser with transform handle gizmos.
    """

    def __init__(
        self,
        server: viser.ViserServer = None,
        minimal: bool = False,
        provide_handles: bool = True,
        device: Literal["cpu", "gpu"] = "cpu",
        left_base_xy: tuple[float, float] = (0.0, 0.0),
        right_base_xy: tuple[float, float] = (0.0, -0.61),  # Hardcoded for YAMS
    ):
        self.minimal = minimal
        self.left_base_xy = left_base_xy
        self.right_base_xy = right_base_xy

        # Set device
        jax.config.update("jax_platform_name", device)

        if not minimal:
            # Initialize viser server
            self.server = server if server is not None else viser.ViserServer()

        # Load robot description using PyRoki for both arms
        try:
            self.urdf = load_robot_description("yam_description")
        except Exception as e:
            print(
                f"Error loading YAM robot description: {e}"
            )
            exit()

        self.robot_left = pk.Robot.from_urdf(self.urdf)
        self.robot_right = pk.Robot.from_urdf(self.urdf)

        # Get rest pose from URDF default configuration
        self.rest_pose = self.urdf.cfg
        self.joints_left = self.rest_pose.copy()
        self.joints_right = self.rest_pose.copy()

        # Target link names for each YAM end effector
        self.target_names = ["link_6"]  # YAM end effector is link_6
        print(f"[YAMSBaseInterface] Target link name for each arm: {self.target_names}")

        if not minimal:
            # Setup visualization
            self._setup_visualization()
            self._setup_gui()
            if provide_handles:
                self._setup_transform_handles()

        # Initialize base poses for each arm
        self.base_pose_left = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([left_base_xy[0], left_base_xy[1], 0.0])
        )
        self.base_pose_right = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([right_base_xy[0], right_base_xy[1], 0.0])
        )

        # Update base frame positions
        if not minimal:
            self.base_frame_left.position = onp.array(self.base_pose_left.translation())
            self.base_frame_left.wxyz = onp.array(self.base_pose_left.rotation().wxyz)
            self.base_frame_right.position = onp.array(self.base_pose_right.translation())
            self.base_frame_right.wxyz = onp.array(self.base_pose_right.rotation().wxyz)

            # Initialize solver parameters
            self.has_jitted_left = False
            self.has_jitted_right = False

        self.print_joint_order()

    def _setup_visualization(self):
        """Setup basic visualization elements for dual YAMs."""
        # Add base frames for both arms
        self.base_frame_left = self.server.scene.add_frame("/base_left", show_axes=False, axes_length=0.2)
        self.base_frame_right = self.server.scene.add_frame("/base_right", show_axes=False, axes_length=0.2)

        # Add robot URDFs for both arms
        self.urdf_vis_left = viser.extras.ViserUrdf(self.server, self.urdf, root_node_name="/base_left")
        self.urdf_vis_right = viser.extras.ViserUrdf(self.server, self.urdf, root_node_name="/base_right")

        # Update initial configurations
        self.urdf_vis_left.update_cfg(self.rest_pose)
        self.urdf_vis_right.update_cfg(self.rest_pose)

        # Add ground grid
        self.server.scene.add_grid("ground", width=4, height=4, cell_size=0.1)

    def _setup_gui(self):
        """Setup GUI elements."""
        # Add timing displays
        self.timing_handle_left = self.server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        self.timing_handle_right = self.server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)

        # Add gizmo size control
        self.tf_size_handle = self.server.gui.add_slider("Gizmo size", min=0.01, max=0.4, step=0.01, initial_value=0.15)

        # Add TCP offset control
        self.use_tcp_offset_handle = self.server.gui.add_checkbox("Use TCP Offset", initial_value=True)

        # Add base position controls
        with self.server.gui.add_folder("Base Positions"):
            self.left_x_handle = self.server.gui.add_slider(
                "Left Base X", min=-2.0, max=2.0, step=0.1, initial_value=self.left_base_xy[0]
            )
            self.left_y_handle = self.server.gui.add_slider(
                "Left Base Y", min=-2.0, max=2.0, step=0.1, initial_value=self.left_base_xy[1]
            )
            self.right_x_handle = self.server.gui.add_slider(
                "Right Base X", min=-2.0, max=2.0, step=0.1, initial_value=self.right_base_xy[0]
            )
            self.right_y_handle = self.server.gui.add_slider(
                "Right Base Y", min=-2.0, max=2.0, step=0.1, initial_value=self.right_base_xy[1]
            )

        # Setup base position update callbacks
        @self.left_x_handle.on_update
        def update_left_x(_):
            self._update_base_positions()

        @self.left_y_handle.on_update
        def update_left_y(_):
            self._update_base_positions()

        @self.right_x_handle.on_update
        def update_right_x(_):
            self._update_base_positions()

        @self.right_y_handle.on_update
        def update_right_y(_):
            self._update_base_positions()

    def _update_base_positions(self):
        """Update base positions based on GUI sliders."""
        # Update base poses
        self.base_pose_left = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([self.left_x_handle.value, self.left_y_handle.value, 0.0])
        )
        self.base_pose_right = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), jnp.array([self.right_x_handle.value, self.right_y_handle.value, 0.0])
        )

        # Update base frame positions
        self.base_frame_left.position = onp.array(self.base_pose_left.translation())
        self.base_frame_right.position = onp.array(self.base_pose_right.translation())

    def _setup_transform_handles(self):
        """Setup transform handles for both YAM end effectors."""
        # Main transform controls for both arms
        self.transform_handles = {
            "left": TransformHandle(
                frame=self.server.scene.add_frame(
                    "/tf_left",
                    axes_length=0.3 * self.tf_size_handle.value,
                    axes_radius=0.01 * self.tf_size_handle.value,
                    origin_radius=0.05 * self.tf_size_handle.value,
                ),
                control=self.server.scene.add_transform_controls(
                    "/base_left/target_left",
                    scale=self.tf_size_handle.value,
                    position=(0.3, 0.0, 0.4),
                    wxyz=(1, 0, 0, 0),
                ),
            ),
            "right": TransformHandle(
                frame=self.server.scene.add_frame(
                    "/tf_right",
                    axes_length=0.3 * self.tf_size_handle.value,
                    axes_radius=0.01 * self.tf_size_handle.value,
                    origin_radius=0.05 * self.tf_size_handle.value,
                ),
                control=self.server.scene.add_transform_controls(
                    "/base_right/target_right",
                    scale=self.tf_size_handle.value,
                    position=(0.3, 0.0, 0.4),
                    wxyz=(1, 0, 0, 0),
                ),
            ),
        }

        # TCP offset frames (180 degrees around x-axis, -0.1m in z)
        self.tcp_frames = {
            "left": self.server.scene.add_frame(
                "/base_left/target_left/tcp_left",
                show_axes=False,
                position=(0.0, 0.0, -0.1),
                # wxyz=(0, 1, 0, 0)  # 180 degrees around x-axis
            ),
            "right": self.server.scene.add_frame(
                "/base_right/target_right/tcp_right",
                show_axes=False,
                position=(0.0, 0.0, -0.1),
                # wxyz=(0, 1, 0, 0)  # 180 degrees around x-axis
            ),
        }

        # Update transform handles when size changes
        @self.tf_size_handle.on_update
        def update_tf_size(_):
            for handle in self.transform_handles.values():
                if handle.control:
                    handle.control.scale = self.tf_size_handle.value
                handle.frame.axes_length = 0.3 * self.tf_size_handle.value
                handle.frame.axes_radius = 0.01 * self.tf_size_handle.value
                handle.frame.origin_radius = 0.05 * self.tf_size_handle.value

    def update_visualization(self):
        """Update visualization with current state for both arms."""
        # Update base frames
        if hasattr(self, "base_frame_left"):
            self.base_frame_left.position = onp.array(self.base_pose_left.translation())
            self.base_frame_left.wxyz = onp.array(self.base_pose_left.rotation().wxyz)
        if hasattr(self, "base_frame_right"):
            self.base_frame_right.position = onp.array(self.base_pose_right.translation())
            self.base_frame_right.wxyz = onp.array(self.base_pose_right.rotation().wxyz)

        # Update robot configurations
        if not self.minimal:
            self.urdf_vis_left.update_cfg(onp.array(self.joints_left))
            self.urdf_vis_right.update_cfg(onp.array(self.joints_right))

        # Update end-effector frames
        if hasattr(self, "transform_handles"):
            # Left arm
            link_poses_left = self.robot_left.forward_kinematics(jnp.array(self.joints_left))
            target_link_idx = self.robot_left.links.names.index(self.target_names[0])
            T_target_world_left = self.base_pose_left @ jaxlie.SE3(link_poses_left[target_link_idx])
            self.transform_handles["left"].frame.position = onp.array(T_target_world_left.translation())
            self.transform_handles["left"].frame.wxyz = onp.array(T_target_world_left.rotation().wxyz)

            # Right arm
            link_poses_right = self.robot_right.forward_kinematics(jnp.array(self.joints_right))
            T_target_world_right = self.base_pose_right @ jaxlie.SE3(link_poses_right[target_link_idx])
            self.transform_handles["right"].frame.position = onp.array(T_target_world_right.translation())
            self.transform_handles["right"].frame.wxyz = onp.array(T_target_world_right.rotation().wxyz)

    def solve_ik(self, target_positions=None, target_wxyzs=None, coordinate_frame: Literal["base", "world"] = "base"):
        """
        Solve inverse kinematics for both YAM arms using PyRoki.

        Args:
            target_positions: Target positions for both arms [left_pos, right_pos] or single position for both
            target_wxyzs: Target orientations (wxyz quaternions) for both arms [left_wxyz, right_wxyz] or single for both
            coordinate_frame: "base" for targets relative to each robot base, "world" for world coordinates
        """
        if not self.minimal and hasattr(self, "transform_handles"):
            # Get target poses from transform controls with TCP offset
            use_tcp_offset = (
                getattr(self.use_tcp_offset_handle, "value", True) if hasattr(self, "use_tcp_offset_handle") else True
            )

            # Process left arm - targets are already in left base frame since controls are parented to /base_left
            left_handle = self.transform_handles["left"]
            if hasattr(self, "tcp_frames") and use_tcp_offset:
                # Compute TCP transform
                handle_transform = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(left_handle.control.wxyz, -1)), left_handle.control.position
                )
                tcp_frame = self.tcp_frames["left"]
                tcp_offset = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(tcp_frame.wxyz, -1)), tcp_frame.position
                )
                tcp_transform = handle_transform @ tcp_offset
                left_target_position = tcp_transform.wxyz_xyz[-3:]
                left_target_wxyz = tcp_transform.wxyz_xyz[:4]
            else:
                left_target_position = left_handle.control.position
                left_target_wxyz = left_handle.control.wxyz

            # Process right arm - targets are already in right base frame since controls are parented to /base_right
            right_handle = self.transform_handles["right"]
            if hasattr(self, "tcp_frames") and use_tcp_offset:
                # Compute TCP transform
                handle_transform = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(right_handle.control.wxyz, -1)), right_handle.control.position
                )
                tcp_frame = self.tcp_frames["right"]
                tcp_offset = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_quaternion_xyzw(onp.roll(tcp_frame.wxyz, -1)), tcp_frame.position
                )
                tcp_transform = handle_transform @ tcp_offset
                right_target_position = tcp_transform.wxyz_xyz[-3:]
                right_target_wxyz = tcp_transform.wxyz_xyz[:4]
            else:
                right_target_position = right_handle.control.position
                right_target_wxyz = right_handle.control.wxyz

        elif target_positions is None or target_wxyzs is None:
            # For minimal mode, use default targets if not provided (in base frame coordinates)
            left_target_position = onp.array([0.3, 0.0, 0.4])
            left_target_wxyz = onp.array([1.0, 0.0, 0.0, 0.0])
            right_target_position = onp.array([0.3, 0.0, 0.4])
            right_target_wxyz = onp.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Process provided targets
            if isinstance(target_positions, (list, tuple)) and len(target_positions) == 2:
                left_target_position = onp.array(target_positions[0])
                right_target_position = onp.array(target_positions[1])
            else:
                # Single target for both arms
                left_target_position = onp.array(target_positions)
                right_target_position = onp.array(target_positions)

            if isinstance(target_wxyzs, (list, tuple)) and len(target_wxyzs) == 2:
                left_target_wxyz = onp.array(target_wxyzs[0])
                right_target_wxyz = onp.array(target_wxyzs[1])
            else:
                # Single orientation for both arms
                left_target_wxyz = onp.array(target_wxyzs)
                right_target_wxyz = onp.array(target_wxyzs)

        # Transform targets from world to base coordinates if needed
        if coordinate_frame == "world":
            # Transform world targets to base frame coordinates
            left_world_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_quaternion_xyzw(onp.roll(left_target_wxyz, -1)), left_target_position
            )
            right_world_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_quaternion_xyzw(onp.roll(right_target_wxyz, -1)), right_target_position
            )

            # Transform to base coordinates
            left_base_pose = self.base_pose_left.inverse() @ left_world_pose
            right_base_pose = self.base_pose_right.inverse() @ right_world_pose

            # Extract position and orientation
            left_target_position = onp.array(left_base_pose.translation())
            left_target_wxyz = onp.array(left_base_pose.rotation().wxyz)
            right_target_position = onp.array(right_base_pose.translation())
            right_target_wxyz = onp.array(right_base_pose.rotation().wxyz)

        # Solve IK for left arm
        timing_start_left = False
        if hasattr(self, "has_jitted_left") and not self.has_jitted_left:
            start_time = time.time()
            timing_start_left = True

        prev_cfg_left = self.joints_left if hasattr(self, "joints_left") else self.rest_pose

        # Targets are already in left robot base frame - no transformation needed
        self.joints_left = solve_ik_with_multiple_targets(
            robot=self.robot_left,
            target_link_names=self.target_names,
            target_positions=onp.array([left_target_position]),
            target_wxyzs=onp.array([left_target_wxyz]),
            prev_cfg=prev_cfg_left,
        )

        if timing_start_left:
            elapsed_time = (time.time() - start_time) * 1000
            if hasattr(self, "timing_handle_left"):
                self.timing_handle_left.value = elapsed_time
            logger.info("Left arm JIT compile + running took {} ms.", elapsed_time)
            self.has_jitted_left = True

        # Solve IK for right arm
        timing_start_right = False
        if hasattr(self, "has_jitted_right") and not self.has_jitted_right:
            start_time = time.time()
            timing_start_right = True

        prev_cfg_right = self.joints_right if hasattr(self, "joints_right") else self.rest_pose

        # Targets are already in right robot base frame - no transformation needed
        self.joints_right = solve_ik_with_multiple_targets(
            robot=self.robot_right,
            target_link_names=self.target_names,
            target_positions=onp.array([right_target_position]),
            target_wxyzs=onp.array([right_target_wxyz]),
            prev_cfg=prev_cfg_right,
        )

        if timing_start_right:
            elapsed_time = (time.time() - start_time) * 1000
            if hasattr(self, "timing_handle_right"):
                self.timing_handle_right.value = elapsed_time
            logger.info("Right arm JIT compile + running took {} ms.", elapsed_time)
            self.has_jitted_right = True

    def home(self):
        """Reset both arms to rest pose."""
        self.joints_left = self.rest_pose.copy()
        self.joints_right = self.rest_pose.copy()

    def print_joint_order(self):
        """
        Print the expected joint order for external interface.

        Since YAM joints are flipped internally, this shows the order that
        external callers should use when providing joint angles to methods
        like compute_forward_kinematics() and update_cfg().
        """
        joint_names_reversed = list(reversed(self.urdf.joint_names))
        print("Expected joint order per arm for external interface (0th is base):")
        for i, joint_name in enumerate(joint_names_reversed):
            print(f"  [{i}]: {joint_name}")
        print(f"Total joints: {len(joint_names_reversed)}")
        return joint_names_reversed

    def get_end_effector_poses(
        self, target_link_names: list[str] | None = None, coordinate_frame: Literal["base", "world"] = "world"
    ) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        """
        Compute forward kinematics and return poses of end effectors for both arms.

        Args:
            target_link_names: List of target link names [left_target, right_target].
                              If None, uses self.target_names or defaults to last link.
            coordinate_frame: "base" for poses relative to each robot base, "world" for world coordinates

        Returns:
            Tuple of (left_ee_pose, right_ee_pose) in requested coordinate frame
        """
        # Determine target link names
        if target_link_names is None:
            if hasattr(self, "target_names") and len(self.target_names) >= 1:
                left_target = self.target_names[0]
                right_target = self.target_names[0]  # Same target for both arms
            else:
                # Fallback to last link
                left_target = self.robot_left.links.names[-1]
                right_target = self.robot_right.links.names[-1]
        else:
            left_target = target_link_names[0] if len(target_link_names) > 0 else self.robot_left.links.names[-1]
            right_target = target_link_names[1] if len(target_link_names) > 1 else target_link_names[0]

        # Left arm FK
        left_link_idx = self.robot_left.links.names.index(left_target)
        link_poses_left = self.robot_left.forward_kinematics(jnp.array(self.joints_left))
        T_base_left_ee = jaxlie.SE3(link_poses_left[left_link_idx])

        # Right arm FK
        right_link_idx = self.robot_right.links.names.index(right_target)
        link_poses_right = self.robot_right.forward_kinematics(jnp.array(self.joints_right))
        T_base_right_ee = jaxlie.SE3(link_poses_right[right_link_idx])

        if coordinate_frame == "world":
            # Transform to world coordinates
            T_world_left_ee = self.base_pose_left @ T_base_left_ee
            T_world_right_ee = self.base_pose_right @ T_base_right_ee
            return T_world_left_ee, T_world_right_ee
        # coordinate_frame == "base"
        # Return poses relative to each robot base
        return T_base_left_ee, T_base_right_ee

    def solve_fk(
        self,
        left_joints: onp.ndarray,
        right_joints: onp.ndarray,
        target_link_names: list[str] | None = None,
        coordinate_frame: Literal["base", "world"] = "world",
    ) -> tuple[jaxlie.SE3, jaxlie.SE3]:
        """
        Compute forward kinematics for given joint configurations.

        Args:
            left_joints: Joint angles for left arm (will be flipped internally for YAM)
            right_joints: Joint angles for right arm (will be flipped internally for YAM)
            target_link_names: List of target link names [left_target, right_target].
                              If None, uses self.target_names or defaults to last link.
            coordinate_frame: "base" for poses relative to each robot base, "world" for world coordinates

        Returns:
            Tuple of (left_ee_pose, right_ee_pose) in requested coordinate frame
        """
        has_batch_axis = len(left_joints.shape) > 1
        # Flip joints for YAM robot configuration
        left_joints_flipped = onp.flip(left_joints, axis=0)
        right_joints_flipped = onp.flip(right_joints, axis=0)

        # Determine target link names
        if target_link_names is None:
            if hasattr(self, "target_names") and len(self.target_names) >= 1:
                left_target = self.target_names[0]
                right_target = self.target_names[0]  # Same target for both arms
            else:
                # Fallback to last link (ON YAMS THIS IS BAD BECAUSE THE LAST LINK IS THE BASE, SO MIGHT NOT BE A GOOD HEURISTIC)
                left_target = self.robot_left.links.names[-1]
                right_target = self.robot_right.links.names[-1]
        else:
            left_target = target_link_names[0] if len(target_link_names) > 0 else self.robot_left.links.names[-1]
            right_target = target_link_names[1] if len(target_link_names) > 1 else target_link_names[0]

        # Left arm FK
        left_link_idx = self.robot_left.links.names.index(left_target)
        link_poses_left = self.robot_left.forward_kinematics(jnp.array(left_joints_flipped))
        if has_batch_axis:
            T_base_left_ee = jaxlie.SE3(link_poses_left[:, left_link_idx])
        else:
            T_base_left_ee = jaxlie.SE3(link_poses_left[left_link_idx])

        # Right arm FK
        right_link_idx = self.robot_right.links.names.index(right_target)
        link_poses_right = self.robot_right.forward_kinematics(jnp.array(right_joints_flipped))
        if has_batch_axis:
            T_base_right_ee = jaxlie.SE3(link_poses_right[:, right_link_idx])
        else:
            T_base_right_ee = jaxlie.SE3(link_poses_right[right_link_idx])

        if coordinate_frame == "world":
            # Transform to world coordinates
            T_world_left_ee = self.base_pose_left @ T_base_left_ee
            T_world_right_ee = self.base_pose_right @ T_base_right_ee
            return T_world_left_ee, T_world_right_ee
        # coordinate_frame == "base"
        # Return poses relative to each robot base
        return T_base_left_ee, T_base_right_ee

    def run(self):
        """Main run loop."""
        while True:
            if not self.minimal:
                self.solve_ik()
            self.update_visualization()
            time.sleep(0.01)

    def update_cfg(self, cfg_left, cfg_right):
        """Update the robot configurations."""
        cfg_left = onp.flip(cfg_left, axis=0)
        cfg_right = onp.flip(cfg_right, axis=0)
        self.joints_left = cfg_left
        self.joints_right = cfg_right
        if not self.minimal:
            self.urdf_vis_left.update_cfg(onp.array(self.joints_left))
            self.urdf_vis_right.update_cfg(onp.array(self.joints_right))

    # Convenience methods for unified interface
    def solve_ik_world(self, target_positions, target_wxyzs=None):
        """Convenience method for IK with world coordinate targets."""
        return self.solve_ik(target_positions, target_wxyzs, coordinate_frame="world")

    def solve_ik_base(self, target_positions, target_wxyzs=None):
        """Convenience method for IK with base coordinate targets."""
        return self.solve_ik(target_positions, target_wxyzs, coordinate_frame="base")

    def solve_fk_world(self, left_joints, right_joints, target_link_names=None):
        """Convenience method for FK returning world coordinates."""
        return self.solve_fk(left_joints, right_joints, target_link_names, coordinate_frame="world")

    def solve_fk_base(self, left_joints, right_joints, target_link_names=None):
        """Convenience method for FK returning base coordinates."""
        return self.solve_fk(left_joints, right_joints, target_link_names, coordinate_frame="base")


"""
Solves IK problem using pyroki.
"""

from collections.abc import Sequence

from jax import Array
import jax_dataclasses as jdc
import jaxlie
from jaxls import Cost
from jaxls import Var
from jaxls import VarValues
import numpy as onp
import pyroki as pk


@Cost.create_factory
def limit_velocity_cost(
    vals: VarValues,
    robot: pk.Robot,
    joint_var: Var[Array],
    prev_cfg: Array,
    dt: float,
    weight: Array | float,
) -> Array:
    """Computes the residual penalizing joint velocity limit violations."""
    joint_vel = (vals[joint_var] - prev_cfg) / dt
    residual = jnp.maximum(0.0, jnp.abs(joint_vel) - robot.joints.velocity_limits)
    return (residual * weight).flatten()


def solve_ik_with_multiple_targets(
    robot: pk.Robot,
    target_link_names: Sequence[str],
    target_wxyzs: onp.ndarray,
    target_positions: onp.ndarray,
    prev_cfg: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_names: Sequence[str]. List of link names to be controlled.
        target_wxyzs: onp.ndarray. Shape: (num_targets, 4). Target orientations.
        target_positions: onp.ndarray. Shape: (num_targets, 3). Target positions.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
        jnp.array(prev_cfg),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    target_joint_indices: jax.Array,
    prev_cfg: jax.Array,
) -> jax.Array:
    JointVar = robot.joint_var_cls

    # Get the batch axes for the variable through the target pose.
    # Batch axes for the variables and cost terms (e.g., target pose) should be broadcastable!
    target_pose = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
    batch_axes = target_pose.get_batch_axes()

    factors = [
        pk.costs.pose_cost_analytic_jac(
            jax.tree.map(lambda x: x[None], robot),
            JointVar(jnp.full(batch_axes, 0)),
            target_pose,
            target_joint_indices,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.rest_cost(
            JointVar(0),
            rest_pose=JointVar.default_factory(),
            weight=1.0,
        ),
        pk.costs.limit_cost(
            robot,
            JointVar(0),
            jnp.array([100.0] * robot.joints.num_joints),
        ),
        limit_velocity_cost(
            robot,
            JointVar(0),
            prev_cfg,
            0.01,  # dt
            10.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [JointVar(0)])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
        )
    )
    return sol[JointVar(0)]


if __name__ == "__main__":
    yams_interface = YAMSBaseInterface(left_base_xy=(-0.6, 0.0), right_base_xy=(0.6, 0.0))
    yams_interface.run()
