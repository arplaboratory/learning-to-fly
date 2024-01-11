import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.crtpstack import CRTPPort
from cflib.crazyflie.commander import SET_SETPOINT_CHANNEL, META_COMMAND_CHANNEL, TYPE_HOVER 
import time
import struct
import argparse
import threading


def send_hover_packet(cf, height, vx=0, vy=0, yawrate=0):
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = SET_SETPOINT_CHANNEL
    pk.data = struct.pack('<Bffff', TYPE_HOVER, vx, vy, yawrate, height)
    cf.send_packet(pk)

def send_learned_policy_packet(cf):
    pk = CRTPPacket()
    pk.port = CRTPPort.COMMANDER_GENERIC
    pk.channel = META_COMMAND_CHANNEL
    pk.data = struct.pack('<B', 1)
    cf.send_packet(pk)

def mode_hover_original(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        prev = current
        send_hover_packet(cf, args.height)

def mode_hover_learned(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.wn", 1)
    set_param(cf, "rlt.target_z", args.height)
    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        prev = current
        send_learned_policy_packet(cf)

def set_param(cf, name, target):
    print(f"Parameter {name} was {cf.param.get_value(name)}, setting to {target}")
    while abs(float(cf.param.get_value(name)) - float(target)) > 1e-5:
        cf.param.set_value(name, target)
        time.sleep(0.1)
    print(f"Parameter {name} is {cf.param.get_value(name)} now")


def mode_trajectory_tracking(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command  (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.wn", 4)
    set_param(cf, "rlt.fei", args.trajectory_interval)
    set_param(cf, "rlt.fes", args.trajectory_scale)
    set_param(cf, "rlt.target_z", args.height)

    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    start_time = time.time()
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        now = time.time()
        if current - prev > 0.1:
            start_time = now
        prev = current

        if now - start_time < args.transition_timeout:
            send_hover_packet(cf, args.height)
        else:
            send_learned_policy_packet(cf)

def mode_takeoff_and_switch(cf, args):
    set_param(cf, "rlt.trigger", 0) # setting the trigger mode to the custom command  (cf. https://github.com/arplaboratory/learning_to_fly_controller/blob/0a7680de591d85813f1cd27834b240aeac962fdd/rl_tools_controller.c#L80)
    set_param(cf, "rlt.wn", 1)
    set_param(cf, "rlt.target_z", 0)

    input("Press enter to start hovering")
    prev = time.time()
    acc = 0
    cnt = 0
    start_time = time.time()
    while True:
        i = input("Hold enter to fly")
        if i == "q":
            break
        current = time.time()
        acc += current - prev
        cnt += 1
        if cnt % 100 == 0:
            print(f"Average rate: {1/(acc / cnt):.3f}Hz")
            acc = 0
            cnt = 0
        now = time.time()
        if current - prev > 0.1:
            start_time = now
        prev = current

        if now - start_time < args.transition_timeout:
            send_hover_packet(cf, args.height)
        else:
            send_learned_policy_packet(cf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_uri = 'radio://0/80/2M/E7E7E7E7E7'
    parser.add_argument('--uri', default=default_uri)
    parser.add_argument('--height', default=0.2, type=float)
    parser.add_argument('--mode', default='hover_learned', choices=['hover_learned', 'hover_original', 'takeoff_and_switch', 'trajectory_tracking'])
    parser.add_argument('--trajectory-scale', default=1, type=float, help="Scale of the trajectory")
    parser.add_argument('--trajectory-interval', default=5.5, type=float, help="Interval of the trajectory")
    parser.add_argument('--transition-timeout', default=3, type=float, help="Time after takeoff with the original controller after which the learned controller is used for trajectory tracking")

    args = parser.parse_args()
    uri = uri_helper.uri_from_env(default=default_uri)
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='/tmp/cf_cache')) as scf:
        if args.mode == "hover_learned":
            mode_hover_learned(scf.cf, args)
        elif args.mode == "hover_original":
            mode_hover_original(scf.cf, args)
        elif args.mode == "takeoff_and_switch":
            mode_takeoff_and_switch(scf.cf, args)
        elif args.mode == "trajectory_tracking":
            mode_trajectory_tracking(scf.cf, args)
        else:
            print("Unknown mode")# 

