import base64
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, cast

from mcp.server.fastmcp import FastMCP, Image
from tools.phosphobot import PhosphoClient
from tools.replay_api import launch_replay

# Object-to-episode mapping
OBJECT_TO_EPISODE = {
    "banana": 0,
    "black circle": 1,
    "green cross": 2,
}

@dataclass
class AppContext:
    phospho: PhosphoClient

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    print("Starting Phosphobot...")
    # start_phosphobot()
    # wait_for_phosphobot()  # Important pour ne pas exécuter les tools trop tôt
    phospho = PhosphoClient()
    try:
        yield AppContext(phospho=phospho)
    finally:
        autostop = os.getenv("PHOSPHOBOT_AUTOSTOP", "0") in ("1", "true", "True")
        if autostop:
            print("Stopping Phosphobot...")
            phospho.stop()
        else:
            print("Leaving Phosphobot running (PHOSPHOBOT_AUTOSTOP disabled)")

# Create server with lifespan
mcp = FastMCP("phospho", lifespan=app_lifespan, dependencies=["requests", "opencv-python-headless", "pillow", "psutil"])


@mcp.tool()
def get_camera_frame() -> Image:
    """
    Retrieve a single frame from the robot camera via the phosphobot API.
    Returns a JPEG image.
    """
    result = PhosphoClient().get("/frames")
    
    if not isinstance(result, dict):
        raise RuntimeError("Invalid response from phosphobot")

    image_b64 = result.get("0") or next(iter(result.values()), None)

    if not image_b64:
        raise RuntimeError("No camera frame returned")

    try:
        return Image(
            data=base64.b64decode(image_b64),
            format="jpeg"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}")

@mcp.tool()
def move_init(robot_id: int | None = None) -> str:
    """
    Initialize the robot (homing/calibration as implemented server-side).
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    app_ctx.phospho.post("/move/init", json={}, params={"robot_id": robot_id} if robot_id is not None else None)
    return "Robot initialized."

@mcp.tool()
def move_absolute(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    rx: float | None = None,
    ry: float | None = None,
    rz: float | None = None,
    open: float | None = None,
    robot_id: int | None = None,
    position_tolerance: float = 1e-2,
    orientation_tolerance: float = 1e-2,
    max_trials: int = 1,
) -> str:
    """
    Move the end-effector to an absolute pose.
    Units: x/y/z in centimeters; rx/ry/rz in degrees. Optional "open" controls the gripper (0..1).
    Tip: call move_init first or use get_robot_status to confirm the robot is initialized.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    payload = {
        "x": x,
        "y": y,
        "z": z,
        "rx": rx,
        "ry": ry,
        "rz": rz,
        "open": open,
        "position_tolerance": position_tolerance,
        "orientation_tolerance": orientation_tolerance,
        "max_trials": max_trials,
    }
    app_ctx.phospho.post("/move/absolute", json=payload, params={"robot_id": robot_id} if robot_id is not None else None)
    return "Move absolute command sent."

@mcp.tool()
def move_relative(
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    rx: float | None = None,
    ry: float | None = None,
    rz: float | None = None,
    open: float | None = None,
    robot_id: int | None = None,
) -> str:
    """
    Move the end-effector by relative deltas.
    Units: x/y/z in centimeters; rx/ry/rz in degrees. Optional "open" controls the gripper (0..1).
    Controls (positive directions):
    - Overall: rz+ rotates left, rz- rotates right; x+ moves forward, x- moves backward; z+ moves up, z- moves down
    - Gripper: ry+ clockwise, ry- counterclockwise; rx+ up, rx- down
    Step rule: each provided component enforces a minimum step of 0.1 (if |v| < 0.1, raise to 0.1 preserving sign).
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    # 强制最小步长为 0.1（单位：cm）。如果提供的增量绝对值非空且小于 0.1，则提升到 0.1，方向保持不变。
    def enforce_step(v: float | None) -> float | None:
        if v is None:
            return None
        if abs(v) < 0.1:
            return 0.1 if v >= 0 else -0.1
        return v

    payload = {
        "x": enforce_step(x),
        "y": enforce_step(y),
        "z": enforce_step(z),
        "rx": enforce_step(rx),
        "ry": enforce_step(ry),
        "rz": enforce_step(rz),
        "open": open,
    }
    app_ctx.phospho.post("/move/relative", json=payload, params={"robot_id": robot_id} if robot_id is not None else None)
    return "Move relative command sent."

@mcp.tool()
def control_gripper(open: float, robot_id: int | None = None) -> str:
    """
    Control the gripper opening. Value in [0, 1], where 0 is closed and 1 is fully open.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    app_ctx.phospho.post(
        "/move/relative",
        json={"open": open},
        params={"robot_id": robot_id} if robot_id is not None else None,
    )
    return "Gripper command sent."

@mcp.tool()
def joints_read(unit: Literal["rad", "motor"] = "rad", joints_ids: list[int] | None = None, robot_id: int | None = None) -> dict:
    """
    Read joint angles.
    - unit: "rad" for radians or "motor" for raw motor units
    - joints_ids: optional list of joint indices to read (reads all if None)
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    res = app_ctx.phospho.post("/joints/read", json={"unit": unit, "joints_ids": joints_ids}, params={"robot_id": robot_id} if robot_id is not None else None)
    return cast(dict, res)

@mcp.tool()
def joints_write(angles: list[float], unit: Literal["rad", "motor"] = "rad", joints_ids: list[int] | None = None, robot_id: int | None = None) -> str:
    """
    Write joint angles.
    - angles: list of target joint values
    - unit: "rad" for radians or "motor" for raw motor units
    - joints_ids: optional list of joint indices to write (writes in order if provided)
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    app_ctx.phospho.post("/joints/write", json={"angles": angles, "unit": unit, "joints_ids": joints_ids}, params={"robot_id": robot_id} if robot_id is not None else None)
    return "Joints write command sent."

@mcp.tool()
def torque_toggle(enabled: bool, robot_id: int | None = None) -> str:
    """
    Enable or disable torque on the robot joints.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    app_ctx.phospho.post("/torque/toggle", json={"torque_status": enabled}, params={"robot_id": robot_id} if robot_id is not None else None)
    return f"Torque {'enabled' if enabled else 'disabled'}."

# @mcp.tool()
# def move_sleep(robot_id: int | None = None) -> str:
#     """
#     Move the robot to a predefined sleep (rest) position.
#     """
#     ctx = mcp.get_context()
#     app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
#     app_ctx.phospho.post("/move/sleep", json={}, params={"robot_id": robot_id} if robot_id is not None else None)
#     return "Move to sleep position requested."

@mcp.tool()
def move_hello() -> str:
    """
    Send a hello greeting via the phosphobot backend.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    launch_replay(episode_id=1, dataset_name="hello", phospho=app_ctx.phospho)
    return "Hello command sent."

@mcp.tool()
def move_confirm() -> str:
    """
    Send a confirm action via the phosphobot backend.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    launch_replay(episode_id=0, dataset_name="hello", phospho=app_ctx.phospho)
    return "Confirm action sent."

@mcp.tool()
def move_rest() -> str:
    """
    Send a rest (sleep) action via the phosphobot backend.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    launch_replay(episode_id=0, dataset_name="rest", phospho=app_ctx.phospho)
    return "Rest action sent."

@mcp.tool()
def move_thinking() -> str:
    """
    Send a thinking action via the phosphobot backend.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    launch_replay(episode_id=0, dataset_name="thinking", phospho=app_ctx.phospho)
    return "Thinking action sent."

@mcp.tool()
def get_robot_status(robot_id: int | None = None) -> dict:
    """
    Get current robot status:
    - initialized: inferred via /end-effector/read (True if available)
    - joints: current joint angles (radians), if available
    - torque_enabled: Unknown (not directly readable)
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)

    params = {"robot_id": robot_id} if robot_id is not None else None

    # 1) 用 /end-effector/read 判断是否初始化（未 init 会返回 400 并提示调用 /move/init）
    initialized = False
    try:
        resp = app_ctx.phospho.post("/end-effector/read", json={"sync": False}, params=params, return_response=True)
        if hasattr(resp, "status_code") and 200 <= resp.status_code < 300:
            initialized = True
        else:
            initialized = False
    except Exception:
        initialized = False

    # 2) 读取关节信息（可选）
    joints = None
    try:
        jr = app_ctx.phospho.post("/joints/read", json={"unit": "rad", "joints_ids": None}, params=params)
        if isinstance(jr, dict) and "angles" in jr:
            joints = jr["angles"]
    except Exception:
        joints = None

    return {
        "initialized": initialized,
        "joints": joints,
        "torque_enabled": "Unknown",
    }

@mcp.tool()
def pickup_object(name: Literal["paper", "pen", "pencil"]) -> str:
    """
    Launch a prerecorded replay that picks up the specified object.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    episode_id = OBJECT_TO_EPISODE.get(name)
    if episode_id is None:
        return f"Unknown object: {name}"
    launch_replay(episode_id=episode_id, dataset_name="mcp-demo", phospho=app_ctx.phospho)
    return f"Launched replay for {name}."


@mcp.tool(name="complete_captcha")
def complete_captcha() -> str:
    """
    Use the arm to complete a CAPTCHA by replaying a recorded episode.
    """
    ctx = mcp.get_context()
    app_ctx = cast(AppContext, ctx.request_context.lifespan_context)
    launch_replay(episode_id=0, dataset_name="enter_captcha", phospho=app_ctx.phospho)
    return "Triggered replay for dataset 'enter_captcha'."
