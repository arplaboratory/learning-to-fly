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

def mode_hover(cf):
    input("Press enter to start hovering")
    while True:
        send_hover_packet(cf, args.height)
        time.sleep(0.05)

def set_param(cf, name, target):
    print(f"Parameter {name} was {cf.param.get_value(name)}, setting to {target}")
    while float(cf.param.get_value(name)) != float(target):
        cf.param.set_value(name, target)
        time.sleep(0.1)
    print(f"Parameter {name} is {cf.param.get_value(name)} now")


def mode_trajectory_tracking(cf, args):
    set_param(cf, "rlt.trigger", 0)
    set_param(cf, "rlt.wn", 4)
    set_param(cf, "rlt.fei", args.trajectory_interval)
    set_param(cf, "rlt.fes", args.trajectory_scale)


    input("Press enter to take off using the original controller")
    def check_for_enter_trajectory_tracking():
        input("Press enter to start tracking the trajectory using the policy")
        stop_thread["stop"] = True

    stop_thread = {"stop": False}
    input_thread = threading.Thread(target=check_for_enter_trajectory_tracking)
    input_thread.start()
    while not stop_thread["stop"]:
        send_hover_packet(cf, args.height)
        time.sleep(0.05)

    def check_for_enter_exit():
        input("Press enter to stop")
        stop_thread["stop"] = True


    stop_thread = {"stop": False}
    input_thread = threading.Thread(target=check_for_enter_exit)
    input_thread.start()
    while stop_thread["stop"]:
        send_hover_packet(cf, args.height)
        send_learned_policy_packet(cf, args.height)
        time.sleep(0.05)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_uri = 'radio://0/80/2M/E7E7E7E7E7'
    parser.add_argument('--uri', default=default_uri)
    parser.add_argument('--height', default=0.2, type=float)
    parser.add_argument('--mode', default='hover')
    parser.add_argument('--trajectory-scale', default=1, type=float)
    parser.add_argument('--trajectory-interval', default=5.5, type=float)

    args = parser.parse_args()
    uri = uri_helper.uri_from_env(default=default_uri)
    cflib.crtp.init_drivers()

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='/tmp/cf_cache')) as scf:
        if args.mode == "hover":
            mode_hover(scf.cf)
        elif args.mode == "trajectory_tracking":
            mode_trajectory_tracking(scf.cf, args)
        else:
            print("Unknown mode")

