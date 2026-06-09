#!/usr/bin/env python3
"""
EGO-Planner 통합 실행 스크립트
offboard → lidar transformer → ego-planner 순서로 실행
Ctrl+C로 전체 종료
"""
import subprocess
import signal
import sys
import os
import time
import threading

CONTAINER = "realgazebo"
ROS_SOURCE = (
    "source /opt/ros/jazzy/setup.bash && "
    "source /home/user/realgazebo/RealGazebo-ROS2/install/setup.bash && "
    "source /home/user/ros2_ws/install/setup.bash"
)
WORK_DIR = "/home/user/ros2_ws/src/ego-planner-a4vai"

_procs = []
_cleaning = False


def docker_exec_bg(cmd, ros_source=True):
    full_cmd = f"{ROS_SOURCE} && {cmd}" if ros_source else cmd
    proc = subprocess.Popen(
        ["docker", "exec", "-u", "user", CONTAINER, "bash", "-c", full_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    _procs.append(proc)
    return proc


def docker_exec(cmd, ros_source=False, timeout=15):
    full_cmd = f"{ROS_SOURCE} && {cmd}" if ros_source else cmd
    try:
        subprocess.run(
            ["docker", "exec", "-u", "user", CONTAINER, "bash", "-c", full_cmd],
            timeout=timeout, capture_output=True,
        )
    except Exception:
        pass


def kill_all():
    docker_exec(
        "ps aux | grep -E 'offboard|ego_logger|pointcloud_transformer|ego_planner|traj_server|rviz' "
        "| grep -v realgazebo | grep -v grep "
        "| awk '{print $2}' | xargs -r kill -9 2>/dev/null; "
        "sleep 1; "
        "ps aux | grep 'ros2 launch ego_planner' | grep -v grep "
        "| awk '{print $2}' | xargs -r kill -9 2>/dev/null",
        timeout=15,
    )


def cleanup(signum=None, frame=None):
    global _cleaning
    if _cleaning:
        print("\n[FORCE EXIT]")
        os._exit(1)
    _cleaning = True
    print("\n\n[CLEANUP] Stopping all processes...")
    for p in _procs:
        try:
            p.kill()
        except Exception:
            pass
    kill_all()
    print("[CLEANUP] Done.")
    os._exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def stream_output(proc, prefix):
    """프로세스 stdout을 실시간 출력"""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"[{prefix}] {line}", end="")
    except Exception:
        pass


def wait_for_keyword(proc, keyword, timeout=60):
    """stdout에서 keyword 등장까지 대기, 그 동안 출력도 표시"""
    start = time.time()
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return False
            continue
        print(line, end="")
        if keyword in line:
            return True
    return False


def main():
    print("=" * 50)
    print("  EGO-Planner Flight Launcher")
    print("=" * 50)

    # 1) 기존 프로세스 정리
    print("\n[1/4] Cleaning up old processes...")
    kill_all()
    time.sleep(1)

    # 2) Offboard 이륙
    print("\n[2/4] Starting offboard takeoff...")
    offboard = docker_exec_bg(f"cd {WORK_DIR} && python3 offboard.py")

    print("  Waiting for takeoff (Ready!)...")
    ok = wait_for_keyword(offboard, "Ready!", timeout=30)
    if not ok:
        print("  [WARN] Takeoff keyword not detected within timeout!")
        print("  Continuing anyway... (check QGC)")
    else:
        print("  Takeoff confirmed!")

    # offboard stdout을 백그라운드에서 계속 출력
    t_off = threading.Thread(target=stream_output, args=(offboard, "offboard"), daemon=True)
    t_off.start()

    # 3) Lidar transformer + ego-planner
    print("\n[3/4] Starting lidar transformer + ego-planner...")
    lidar = docker_exec_bg(f"cd {WORK_DIR} && python3 pointcloud_transformer_fast.py")
    t_lidar = threading.Thread(target=stream_output, args=(lidar, "lidar"), daemon=True)
    t_lidar.start()
    time.sleep(2)

    # 4) EGO-Planner
    ego = docker_exec_bg("ros2 launch ego_planner airsim_px4.launch.py")

    print("  Waiting for ego-planner (WAIT_TARGET)...")
    ok = wait_for_keyword(ego, "WAIT_TARGET", timeout=60)
    if ok:
        print("  EGO-Planner ready!")
    else:
        print("  [WARN] WAIT_TARGET not detected, ego-planner may still be loading...")

    t_ego = threading.Thread(target=stream_output, args=(ego, "ego"), daemon=True)
    t_ego.start()

    # 5) RViz
    print("\n[4/4] Starting RViz...")
    rviz = docker_exec_bg("ros2 launch ego_planner rviz.launch.py")
    t_rviz = threading.Thread(target=stream_output, args=(rviz, "rviz"), daemon=True)
    t_rviz.start()

    print("\n" + "=" * 50)
    print("  All nodes running! Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    # 메인 스레드 대기
    while True:
        # 프로세스 상태 체크
        for name, proc in [("offboard", offboard), ("lidar", lidar), ("ego-planner", ego), ("rviz", rviz)]:
            if proc.poll() is not None:
                print(f"\n[WARN] {name} exited (code={proc.returncode})")
        time.sleep(5)


if __name__ == "__main__":
    main()
