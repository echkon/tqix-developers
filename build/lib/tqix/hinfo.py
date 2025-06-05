"""
>>> this file is a part of tqix: a Toolbox for Quantum in X
                              x: quantum measurement, quantum metrology, 
                                 quantum tomography, and more.
________________________________
>>> copyright (c) 2019 and later
>>> authors: Binho Le
>>> all rights reserved
________________________________
"""

__all__ = ['hinfo']

import os
import sys
import multiprocessing
import numpy as np

def _mac_hardware_info():
    info = dict()
    results = dict()
    for l in [l.split(':') for l in os.popen('sysctl hw').readlines()[1:]]:
        info[l[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = \
            l[1].strip('.\n ')
    results.update({'cpus': int(info['physicalcpu'])})
    results.update({'cpu_freq': int(float(os.popen('sysctl -n machdep.cpu.brand_string')
                                .readlines()[0].split('@')[1][:-4])*1000)})
    results.update({'memsize': int(int(info['memsize']) / (1024 ** 2))})
    # add OS information
    results.update({'os': 'Mac OSX'})
    return results


def _linux_hardware_info():
    results = {}
    # get cpu number
    sockets = 0
    cores_per_socket = 0
    frequency = 0.0
    for l in [l.split(':') for l in open("/proc/cpuinfo").readlines()]:
        if (l[0].strip() == "physical id"):
            sockets = np.maximum(sockets,int(l[1].strip())+1)
        if (l[0].strip() == "cpu cores"):
            cores_per_socket = int(l[1].strip())
        if (l[0].strip() == "cpu MHz"):
            frequency = float(l[1].strip()) / 1000.
    results.update({'cpus': sockets * cores_per_socket})
    # get cpu frequency directly (bypasses freq scaling)
    try:
        file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        line = open(file).readlines()[0]
        frequency = float(line.strip('\n')) / 1000000.
    except:
        pass
    results.update({'cpu_freq': frequency})

    # get total amount of memory
    mem_info = dict()
    for l in [l.split(':') for l in open("/proc/meminfo").readlines()]:
        mem_info[l[0]] = l[1].strip('.\n ').strip('kB')
    results.update({'memsize': int(mem_info['MemTotal']) / 1024})
    # add OS information
    results.update({'os': 'Linux'})
    return results

def _freebsd_hardware_info():
    results = {}
    results.update({'cpus': int(os.popen('sysctl -n hw.ncpu').readlines()[0])})
    results.update({'cpu_freq': int(os.popen('sysctl -n dev.cpu.0.freq').readlines()[0])})
    results.update({'memsize': int(os.popen('sysctl -n hw.realmem').readlines()[0]) / 1024})
    results.update({'os': 'FreeBSD'})
    return results

def _win_hardware_info():
    try:
        from comtypes.client import CoGetObject
        winmgmts_root = CoGetObject("winmgmts:root\cimv2")
        cpus = winmgmts_root.ExecQuery("Select * from Win32_Processor")
        ncpus = 0
        for cpu in cpus:
            ncpus += int(cpu.Properties_['NumberOfCores'].Value)
    except:
        ncpus = int(multiprocessing.cpu_count())
    return {'os': 'Windows', 'cpus': ncpus}


def hinfo():
    """
    Returns basic hardware information about the computer.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns
    -------
    info : dict
        Dictionary containing cpu and memory information.

    """
    if sys.platform == 'darwin':
        out = _mac_hardware_info()
    elif sys.platform == 'win32':
        out = _win_hardware_info()
    elif sys.platform in ['linux', 'linux2']:
        out = _linux_hardware_info()
    elif sys.platform.startswith('freebsd'):
        out = _freebsd_hardware_info()
    else:
        out = {}
    return out

if __name__ == '__main__':
    print(hinfo())
