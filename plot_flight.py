#!/usr/bin/env python3
"""
EGO-Planner Flight Data Plotter
사용법: python3 plot_flight.py [csv파일경로]
       csv파일 없이 실행하면 logs/ 폴더의 최신 파일 사용
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    print(f'Loaded {len(data)} rows from {os.path.basename(path)}')
    return data


def find_latest_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs')
    files = sorted(glob.glob(os.path.join(log_dir, 'ego_flight_20260228_112258.csv')))
    if not files:
        print('No CSV files found in logs/')
        sys.exit(1)
    return files[-1]


def plot_all(data):
    t = data['time_sec'].astype(float)

    # ego target (ENU)
    ego_x = data['ego_pos_x'].astype(float)
    ego_y = data['ego_pos_y'].astype(float)
    ego_z = data['ego_pos_z'].astype(float)
    ego_vx = data['ego_vel_x'].astype(float)
    ego_vy = data['ego_vel_y'].astype(float)
    ego_vz = data['ego_vel_z'].astype(float)
    ego_yaw = np.degrees(data['ego_yaw'].astype(float))

    # actual (ENU)
    act_x = data['actual_pos_x'].astype(float)
    act_y = data['actual_pos_y'].astype(float)
    act_z = data['actual_pos_z'].astype(float)
    act_vx = data['actual_vel_x'].astype(float)
    act_vy = data['actual_vel_y'].astype(float)
    act_vz = data['actual_vel_z'].astype(float)
    act_yaw_raw = np.degrees(data['actual_yaw'].astype(float))
    act_yaw = np.degrees(np.arctan2(np.sin(np.radians(act_yaw_raw + 90)), np.cos(np.radians(act_yaw_raw + 90))))  # +90 offset, wrap to [-180,180]

    # error
    err_x = data['error_pos_x'].astype(float)
    err_y = data['error_pos_y'].astype(float)
    err_z = data['error_pos_z'].astype(float)
    err_norm = data['error_pos_norm'].astype(float)

    # replan events
    replan = data['replan'].astype(int)
    replan_times = t[replan == 1]

    mean_err = np.mean(err_norm)

    # ---------- Figure 1: Trajectory + Position + Error ----------
    fig1 = plt.figure(figsize=(16, 14))
    fig1.suptitle('EGO-Planner Flight Analysis', fontsize=14, fontweight='bold')
    gs1 = GridSpec(3, 2, figure=fig1, hspace=0.35, wspace=0.3)

    # 1-1: XY trajectory (top-down)
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(ego_x, ego_y, 'b-', linewidth=0.8, label='Target', alpha=0.7)
    ax1.plot(act_x, act_y, 'r-', linewidth=0.8, label='Actual', alpha=0.7)
    ax1.plot(act_x[0], act_y[0], 'go', markersize=8, label='Start')
    ax1.plot(act_x[-1], act_y[-1], 'rs', markersize=8, label='End')
    for rt in replan_times:
        idx = np.argmin(np.abs(t - rt))
        ax1.plot(act_x[idx], act_y[idx], 'k^', markersize=4, alpha=0.5)
    ax1.set_xlabel('ENU X (East) [m]')
    ax1.set_ylabel('ENU Y (North) [m]')
    ax1.set_title('XY Trajectory (Top-Down)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 1-2: 3D trajectory
    ax2 = fig1.add_subplot(gs1[0, 1], projection='3d')
    ax2.plot(ego_x, ego_y, ego_z, 'b-', linewidth=0.8, label='Target', alpha=0.7)
    ax2.plot(act_x, act_y, act_z, 'r-', linewidth=0.8, label='Actual', alpha=0.7)
    ax2.set_xlabel('X (East)')
    ax2.set_ylabel('Y (North)')
    ax2.set_zlabel('Z (Up)')
    z_mid = (np.min(np.concatenate([ego_z, act_z])) + np.max(np.concatenate([ego_z, act_z]))) / 2
    ax2.set_zlim(z_mid - 5, z_mid + 5)
    ax2.set_title('3D Trajectory')
    ax2.legend(fontsize=8)

    # 1-3: X position tracking
    ax3 = fig1.add_subplot(gs1[1, 0])
    ax3.plot(t, ego_x, 'b--', linewidth=0.8, label='Target X', alpha=0.7)
    ax3.plot(t, act_x, 'b-', linewidth=0.8, label='Actual X')
    for rt in replan_times:
        ax3.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('X Position [m]')
    ax3.set_title('X (East) Position Tracking')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 1-4: Y position tracking
    ax4 = fig1.add_subplot(gs1[1, 1])
    ax4.plot(t, ego_y, 'r--', linewidth=0.8, label='Target Y', alpha=0.7)
    ax4.plot(t, act_y, 'r-', linewidth=0.8, label='Actual Y')
    for rt in replan_times:
        ax4.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Y Position [m]')
    ax4.set_title('Y (North) Position Tracking')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 1-5: Z position tracking
    ax5 = fig1.add_subplot(gs1[2, 0])
    ax5.plot(t, ego_z, 'g--', linewidth=0.8, label='Target Z', alpha=0.7)
    ax5.plot(t, act_z, 'g-', linewidth=0.8, label='Actual Z')
    for rt in replan_times:
        ax5.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Altitude [m]')
    ax5.set_title('Z (Up) Position Tracking')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 1-6: Position error (per axis)
    ax6 = fig1.add_subplot(gs1[2, 1])
    ax6.plot(t, err_x, 'b-', linewidth=0.8, label='Error X', alpha=0.7)
    ax6.plot(t, err_y, 'r-', linewidth=0.8, label='Error Y', alpha=0.7)
    ax6.plot(t, err_z, 'g-', linewidth=0.8, label='Error Z', alpha=0.7)
    for rt in replan_times:
        ax6.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Error [m]')
    ax6.set_title('Position Error (per axis)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Roll / Pitch (if available)
    has_attitude = 'actual_roll' in data.dtype.names
    if has_attitude:
        act_roll = np.degrees(data['actual_roll'].astype(float))
        act_pitch = np.degrees(data['actual_pitch'].astype(float))

    # ---------- Figure 2: Velocity + Yaw + Attitude ----------
    fig2 = plt.figure(figsize=(16, 12))
    fig2.suptitle('Velocity & Attitude', fontsize=14, fontweight='bold')
    gs2 = GridSpec(3, 2, figure=fig2, hspace=0.35, wspace=0.3)

    # 2-1: Velocity X
    ax7 = fig2.add_subplot(gs2[0, 0])
    ax7.plot(t, ego_vx, 'b--', linewidth=0.8, label='Target Vx', alpha=0.7)
    ax7.plot(t, act_vx, 'b-', linewidth=0.8, label='Actual Vx')
    for rt in replan_times:
        ax7.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Velocity [m/s]')
    ax7.set_title('Velocity X (East)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 2-2: Velocity Y
    ax8 = fig2.add_subplot(gs2[0, 1])
    ax8.plot(t, ego_vy, 'r--', linewidth=0.8, label='Target Vy', alpha=0.7)
    ax8.plot(t, act_vy, 'r-', linewidth=0.8, label='Actual Vy')
    for rt in replan_times:
        ax8.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Velocity [m/s]')
    ax8.set_title('Velocity Y (North)')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 2-3: Velocity Z
    ax9 = fig2.add_subplot(gs2[1, 0])
    ax9.plot(t, ego_vz, 'g--', linewidth=0.8, label='Target Vz', alpha=0.7)
    ax9.plot(t, act_vz, 'g-', linewidth=0.8, label='Actual Vz')
    for rt in replan_times:
        ax9.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Velocity [m/s]')
    ax9.set_title('Velocity Z (Up)')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 2-4: Yaw tracking
    ax10 = fig2.add_subplot(gs2[1, 1])
    ax10.plot(t, ego_yaw, 'b--', linewidth=0.8, label='Target Yaw', alpha=0.7)
    ax10.plot(t, act_yaw, 'r-', linewidth=0.8, label='Actual Yaw')
    for rt in replan_times:
        ax10.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
    ax10.set_xlabel('Time [s]')
    ax10.set_ylabel('Yaw [deg]')
    ax10.set_title('Yaw Tracking')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)

    # 2-5: Roll
    if has_attitude:
        ax11 = fig2.add_subplot(gs2[2, 0])
        ax11.plot(t, act_roll, 'b-', linewidth=0.8, label='Roll')
        for rt in replan_times:
            ax11.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
        ax11.set_xlabel('Time [s]')
        ax11.set_ylabel('Roll [deg]')
        ax11.set_title('Roll')
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)

        # 2-6: Pitch
        ax12 = fig2.add_subplot(gs2[2, 1])
        ax12.plot(t, act_pitch, 'r-', linewidth=0.8, label='Pitch')
        for rt in replan_times:
            ax12.axvline(x=rt, color='gray', linewidth=0.5, alpha=0.3)
        ax12.set_xlabel('Time [s]')
        ax12.set_ylabel('Pitch [deg]')
        ax12.set_title('Pitch')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)

    # ---------- Summary stats ----------
    print('\n===== Flight Summary =====')
    flight_time = t[-1] - t[0]
    print(f'Flight time:    {flight_time:.1f} s')
    print(f'Total replans:  {int(replan.sum())}')
    print(f'Mean pos error: {mean_err:.3f} m')
    print(f'Max pos error:  {np.max(err_norm):.3f} m')
    print(f'Mean velocity:  {np.mean(np.sqrt(act_vx**2 + act_vy**2 + act_vz**2)):.2f} m/s')
    print(f'Max velocity:   {np.max(np.sqrt(act_vx**2 + act_vy**2 + act_vz**2)):.2f} m/s')

    plt.show()


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_csv()

    print(f'Using: {csv_path}')
    data = load_csv(csv_path)
    plot_all(data)


if __name__ == '__main__':
    main()
