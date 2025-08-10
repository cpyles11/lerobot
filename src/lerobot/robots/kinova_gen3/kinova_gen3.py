import logging
import threading
import time
from functools import cached_property
from typing import Any

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots import Robot
from lerobot.robots.kinova_gen3.config_kinova_gen3 import KinovaGen3Config


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            e.set()

    return check


logger = logging.getLogger(__name__)


class KinovaGen3(Robot):
    config_class = KinovaGen3Config
    name = "kinova_gen3"

    def __init__(self, config: KinovaGen3Config):
        super().__init__(config)
        self.config = config

        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

        self.base: BaseClient | None = None
        self.session_manager: SessionManager | None = None
        self.device_config: DeviceConfigClient | None = None
        self.device_manager: DeviceManagerClient | None = None
        self.vision_config: VisionConfigClient | None = None

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        # TODO May need to change to not rely on a connection happening first...
        # Joint positions
        measured_joint_angles = self.base.GetMeasuredJointAngles()
        obs_dict = {
            f"joint_id_{joint_angle.joint_identifier}.pos": float
            for joint_angle in measured_joint_angles.joint_angles
        }
        # gripper position
        obs_dict["gripper.post"] = float
        return obs_dict

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        # TODO: Make this better, not sure this is verifying we can talk to self.base

        return self.transport.tcp_transport_thread.is_alive() and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self):

        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.transport.connect(self.config.ip, self.config.port)

        if self.config.kinova_user != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.config.kinova_user
            session_info.password = self.config.kinova_pwd
            session_info.session_inactivity_timeout = (
                self.config.session_inactivity_timeout
            )  # (milliseconds)
            session_info.connection_inactivity_timeout = (
                self.config.connection_inactivity_timeout
            )  # (milliseconds)

            self.session_manager = SessionManager(self.router)
            print("Logging as", self.config.kinova_user, "on device", self.config.ip)
            self.session_manager.CreateSession(session_info)

        self.base = BaseClient(self.router)
        self.device_config = DeviceConfigClient(self.router)
        self.device_manager = DeviceManagerClient(self.router)
        self.vision_config = VisionConfigClient(self.router)

        for cam in self.cameras.values():
            if cam.config.type == "kinova_gen3_vision":
                cam.connect(self.vision_config, self.device_manager)
            else:
                cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def get_observation(self) -> dict[str, Any]:

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        measured_joint_angles = self.base.GetMeasuredJointAngles()
        obs_dict = {
            f"joint_id_{joint_angle.joint_identifier}.pos": joint_angle.value
            for joint_angle in measured_joint_angles.joint_angles
        }

        # Read gripper position
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        measured_gripper_position = self.base.GetMeasuredGripperMovement(
            gripper_request
        )
        obs_dict["gripper.post"] = measured_gripper_position.finger[0].value

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @property
    def is_calibrated(self) -> bool:
        """Whether the robot is currently calibrated or not. Should be always `True` if not applicable"""
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot if applicable. If not, this should be a no-op.

        This method should collect any necessary data (e.g., motor offsets) and update the
        :pyattr:`calibration` dictionary accordingly.
        """
        pass

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.

        For now, relying on user to set and maintain consisten configs, i.e. via web gui, other than camera FPS and resolution
        """
        # TODO:  send to initial position (there should be examples which address this)

        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    def move_to_start_position(self):
        # Move arm to ready position
        constrained_joint_angles = Base_pb2.ConstrainedJointAngles()

        actuator_count = self.base.GetActuatorCount().count
        angles = [0.0] * actuator_count

        # Actuator 4 at 90 degrees
        for joint_id in range(len(angles)):
            joint_angle = constrained_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        print("Reaching joint angles...")
        self.base.PlayJointTrajectory(constrained_joint_angles)

        print("Waiting for movement to finish ...")
        finished = e.wait(self.config.action_timeout_duration)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Joint angles reached")
        else:
            print("Timeout on action notification wait")
        return finished

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`action_features`.

        Returns:
            dict[str, Any]: The action actually sent to the motors potentially clipped or modified, e.g. by
                safety limits on velocity.
        """
        pass

    def disconnect(self) -> None:
        """Disconnect from the robot and perform any necessary cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        if self.session_manager is not None:

            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.session_manager.CloseSession(router_options)

        self.transport.disconnect()

        logger.info(f"{self} disconnected.")

    def example_send_joint_speeds(self):

        joint_speeds = Base_pb2.JointSpeeds()

        actuator_count = self.base.GetActuatorCount().count
        # The 7DOF robot will spin in the same direction for 10 seconds
        if actuator_count == 7:
            speeds = [
                self.config.speed,
                0,
                -self.config.speed,
                0,
                self.config.speed,
                0,
                -self.config.speed,
            ]
            i = 0
            for speed in speeds:
                joint_speed = joint_speeds.joint_speeds.add()
                joint_speed.joint_identifier = i
                joint_speed.value = speed
                joint_speed.duration = 0
                i = i + 1
            print("Sending the joint speeds for 10 seconds...")
            self.base.SendJointSpeedsCommand(joint_speeds)
            time.sleep(10)
        # The 6 DOF robot will alternate between 4 spins, each for 2.5 seconds
        if actuator_count == 6:
            print("Sending the joint speeds for 10 seconds...")
            for times in range(4):
                del joint_speeds.joint_speeds[:]
                if times % 2:
                    speeds = [-self.config.speed, 0.0, 0.0, self.config.speed, 0.0, 0.0]
                else:
                    speeds = [self.config.speed, 0.0, 0.0, -self.config.speed, 0.0, 0.0]
                i = 0
                for speed in speeds:
                    joint_speed = joint_speeds.joint_speeds.add()
                    joint_speed.joint_identifier = i
                    joint_speed.value = speed
                    joint_speed.duration = 0
                    i = i + 1

                self.base.SendJointSpeedsCommand(joint_speeds)
                time.sleep(2.5)

        print("Stopping the robot")
        self.base.Stop()

        return True
