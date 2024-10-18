import sys
import time
import random
import MDSplus
# from mdsthin import MDSplus
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
all_shots = [int(line.rstrip()) for line in open('disruption_warnings_all_shots.csv')]

class SigObj:
   def __init__(self, sig_name, tree, signal):
      self.sig_name = sig_name.lower()
      self.tree = tree.lower()
      self.signal = signal.lower()


# Use "raw" strings when entering tag names (so don't have to enter "\\" for "\").
# These four signals were used for initial tests of the program.
#            SigObj("ip", "cmod", r"\Ip"),
#            SigObj("efit_aminor", "cmod", r"\efit_aeqdsk:aminor"), 
#            SigObj("bolo_bright", "spectroscopy", "BOLOMETER.RESULTS.DIODE.AXA:BRIGHT"),
#            SigObj("xtomo_chord16", "xtomo", "BRIGHTNESSES.ARRAY_1:CHORD_16")

# These ~70 signals are most of those that disruption-py retrieves by default.
signals = [ 
            SigObj("mflux_v0", "analysis", r"\top.mflux:v0"),
            SigObj("aeqdsk_time", "cmod", r"\analysis::efit_aeqdsk:time"),
            SigObj("aeqdsk_aminor", "cmod", r"\efit_aeqdsk:aminor"),
            SigObj("aeqdsk_area", "cmod", r"\efit_aeqdsk:area"),
            SigObj("aeqdsk_betan", "cmod", r"\efit_aeqdsk:betan"),
            SigObj("aeqdsk_betap", "cmod", r"\efit_aeqdsk:betap"),
            SigObj("aeqdsk_chisq", "cmod", r"\efit_aeqdsk:chisq"),
            SigObj("aeqdsk_doutl", "cmod", r"\efit_aeqdsk:doutl"),
            SigObj("aeqdsk_doutu", "cmod", r"\efit_aeqdsk:doutu"),
            SigObj("aeqdsk_eout", "cmod", r"\efit_aeqdsk:eout"),
            SigObj("aeqdsk_kappa", "cmod", r"\efit_aeqdsk:kappa"),
            SigObj("aeqdsk_li", "cmod", r"\efit_aeqdsk:li"),
            SigObj("aeqdsk_obott", "cmod", r"\efit_aeqdsk:obott/100"),
            SigObj("aeqdsk_otop", "cmod", r"\efit_aeqdsk:otop/100"),
            SigObj("aeqdsk_q0", "cmod", r"\efit_aeqdsk:q0"),
            SigObj("aeqdsk_q95", "cmod", r"\efit_aeqdsk:q95"),
            SigObj("aeqdsk_qstar", "cmod", r"\efit_aeqdsk:qstar"),
            SigObj("aeqdsk_rmagx", "cmod", r"\efit_aeqdsk:rmagx"),
            SigObj("aeqdsk_ssep", "cmod", r"\efit_aeqdsk:ssep/100"),
            SigObj("aeqdsk_time2", "cmod", r"\efit_aeqdsk:time"),
            SigObj("aeqdsk_vloopt", "cmod", r"\efit_aeqdsk:vloopt"),
            SigObj("aeqdsk_wplasm", "cmod", r"\efit_aeqdsk:wplasm"),
            SigObj("aeqdsk_xnnc", "cmod", r"-\efit_aeqdsk:xnnc"),
            SigObj("aeqdsk_zmagx", "cmod", r"\efit_aeqdsk:zmagx"),
            SigObj("aeqdsk_ssibry", "cmod", r"\efit_geqdsk:ssibry"),
            SigObj("electrons_fiberz", "electrons", r"\fiber_z"),
            SigObj("electrons_nl04", "electrons", r"\electrons::top.tci.results:nl_04"),
            SigObj("electrons_te_rz", "electrons", r"\electrons::top.yag_new.results.profiles:te_rz"),
            SigObj("electrons_zsrt", "electrons", r"\electrons::top.yag_new.results.profiles:z_sorted"),
            SigObj("eng_r_cur", "engineering", r"\efc:u_bus_r_cur"),
            SigObj("hybrid_s1p2_factor", "dpcs", r"\dpcs::top.seg_01:p_02:predictor:factor"),
            SigObj("hybrid_s2p2_factor", "dpcs", r"\dpcs::top.seg_02:p_02:predictor:factor"),
            SigObj("hybrid_in056_p2v", "hybrid", r"\hybrid::top.dpcs_config.inputs:input_056:p_to_v_expr"),
            SigObj("hybrid_in056", "hybrid", r"\hybrid::top.hardware.dpcs.signals.a_in:input_056"),
            SigObj("hybrid_aout", "hybrid", r"\top.hardware.dpcs.signals:a_out"),
            SigObj("lh_netpow", "LH", r"\LH::TOP.RESULTS:NETPOW"),
            SigObj("mag_btor", "magnetics", r"\btor"),
            SigObj("mag_ip", "magnetics", r"\ip"),
            SigObj("mag_pickup", "magnetics", r"\mag_bp_coils.btor_pickup"),
            SigObj("mag_nodename", "magnetics", r"\mag_bp_coils.nodename"),
            SigObj("mag_phi", "magnetics", r"\mag_bp_coils.phi"),
            SigObj("mag_bp13bc", "magnetics", r"\mag_bp_coils.signals.BP13_BC"),
            SigObj("mag_bp13de", "magnetics", r"\mag_bp_coils.signals.BP13_DE"),
            SigObj("mag_bp13gh", "magnetics", r"\mag_bp_coils.signals.BP13_GH"),
            SigObj("mag_bp13jk", "magnetics", r"\mag_bp_coils.signals.BP13_JK"),
            SigObj("cmod_ts_te", "cmod", r"\ts_te"),
            SigObj("pcs_s1p01_name", "pcs", r"\PCS::TOP.SEG_01:P_01:name"),
            SigObj("pcs_s1p02", "pcs", r"\PCS::TOP.SEG_01:P_02"),
            SigObj("pcs_s1p02_name", "pcs", r"\PCS::TOP.SEG_01:P_02:name"),
            SigObj("pcs_s1p02_pid", "pcs", r"\PCS::TOP.SEG_01:P_02:pid_gains"),
            SigObj("pcs_s1p16_name", "pcs", r"\PCS::TOP.SEG_01:P_16:name"),
            SigObj("pcs_s1p16_pid", "pcs", r"\PCS::TOP.SEG_01:P_16:pid_gains"),
            SigObj("pcs_s1start", "pcs", r"\PCS::TOP.SEG_01:start_time"),
            SigObj("pcs_s2p01_name", "pcs", r"\PCS::TOP.SEG_02:P_01:name"),
            SigObj("pcs_s2p02", "pcs", r"\PCS::TOP.SEG_02:P_02"),
            SigObj("pcs_s2p02_name", "pcs", r"\PCS::TOP.SEG_02:P_02:name"),
            SigObj("pcs_s2p02_pid", "pcs", r"\PCS::TOP.SEG_02:P_02:pid_gains"),
            SigObj("pcs_s2p16", "pcs", r"\PCS::TOP.SEG_02:P_16"),
            SigObj("pcs_s2p16_name", "pcs", r"\PCS::TOP.SEG_02:P_16:name"),
            SigObj("pcs_s2p16_pid", "pcs", r"\PCS::TOP.SEG_02:P_16:pid_gains"),
            SigObj("pcs_s2start", "pcs", r"\PCS::TOP.SEG_02:start_time"),
            SigObj("rf_power", "RF", r"\rf::rf_power_net"),
            SigObj("bolo_axa_good", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.DIODE_CALIB.AXA:GOOD"),
            SigObj("bolo_axa_zo", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.DIODE_CALIB.AXA:Z_O"),
            SigObj("bolo_axj_good", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.DIODE_CALIB.AXJ:GOOD"),
            SigObj("bolo_axj_zo", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.DIODE_CALIB.AXJ:Z_O"),
            SigObj("bolo_axa_bright", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.AXA:BRIGHT"),
            SigObj("bolo_axj_bright", "spectroscopy", r"\SPECTROSCOPY::TOP.BOLOMETER.RESULTS.DIODE.AXJ:BRIGHT"),
            SigObj("spect_twopi", "spectroscopy", r"\twopi_diode"),
            SigObj("xtomo_chord16", "xtomo", r"\top.brightnesses.array_1:chord_16")
         ]

def hdf_bench(shots, sigs):
    import h5py
    for shot in shots:
        try:
            with h5py.File(f'/home/jas/benchmarks/hdf/{shot}.hdf5', 'r') as f:
                for s in sigs:
                    try:
                        y = f[s.signal][:]
                        x = f[f'{s.signal}_time'][:]
                    except Exception as e:
                        pass
                        # print(f'Error reading {s.signal} from shot {shot}')
        except Exception as e:
            print(f'error opening shot {shot}\n{e}')    

def distributed_bench(shots, sigs):
    import MDSplus
    for shot in shots:
        try:
            tree = MDSplus.Tree('cmod', shot)
            for s in sigs:
                try:
                    sig = tree.getNode(s.signal)
                    y = sig.data()
                    x = sig.dim_of().data()
                except Exception as e:
                    pass
                #     print(f'Error reading {s.signal} from shot {shot}')
        except Exception as e:
            print(f'error opening shot {shot}\n{e}')

def thin_bench(shots, sigs):
    c = MDSplus.Connection('alcdata-archives')
    dummy = c.get('setenv("PyLib=python2.7")')
    dummy = c.get('shorten_path()')
    for shot in shots:
        c.openTree('cmod', shot)
        for s in sigs:
            try:
                y = c.get(f'_sig = {s.signal}')
                x = c.get('dim_of(_sig)')
            except Exception as e:
                # print(f'could not read {s.signal} from {shot}')
                # print(e)
                pass

def gm_bench(shots, sigs):
    c = MDSplus.Connection('alcdata-archives')
    dummy = c.get('setenv("PyLib=python2.7")')
    dummy = c.get('shorten_path()')
    gm = c.getMany()
    for s in sigs:
        gm.append(s.sig_name,f'_sig={s.signal}')
        gm.append(f'{s.sig_name}_time', 'dim_of(_sig)')   
    for shot in shots:
        ans = gm.execute()
        for s in sigs:
            try:
                y = ans[s.sig_name]
                x = ans[f'{s.sig_name}_time']
            except Exception as e:
                print(f'could not read {s.signal} from {shot}')
                print(e)

def THIN_BENCH(shots, sigs, threads):
    # Handle edge case where number of threads exceeds available data
    chunk_size = len(shots) // threads
    remainder = len(shots) % threads

    # If the number of threads does not evenly divide the array, ensure the chunks are sized properly
    if remainder != 0:
        shots = shots[:-(remainder)]  # Remove remainder to get equal-sized chunks

    # Reshape the integer array into chunks
    int_chunks = np.array(shots).reshape(threads, chunk_size)

    # Set up multiprocessing pool
    start = time.time()
    with Pool(threads) as pool:
        # Using starmap to send both the int_chunks and the full string array to the function
        results = pool.starmap(thin_bench, [(chunk, sigs) for chunk in int_chunks])

    print(f'Thin {len(shots)} shots {len(sigs)} signals {threads} threads {time.time()-start}) seconds')
    return time.time() - start

def GM_BENCH(shots, sigs, threads):
    # Handle edge case where number of threads exceeds available data
    chunk_size = len(shots) // threads
    remainder = len(shots) % threads

    # If the number of threads does not evenly divide the array, ensure the chunks are sized properly
    if remainder != 0:
        shots = shots[:-(remainder)]  # Remove remainder to get equal-sized chunks

    # Reshape the integer array into chunks
    int_chunks = np.array(shots).reshape(threads, chunk_size)

    # Set up multiprocessing pool
    start = time.time()
    with Pool(threads) as pool:
        # Using starmap to send both the int_chunks and the full string array to the function
        results = pool.starmap(gm_bench, [(chunk, sigs) for chunk in int_chunks])

    print(f'GetMany {len(shots)} shots {len(sigs)} signals {threads} threads {time.time()-start}) seconds')
    return time.time() - start

def DISTRIBUTED_BENCH(shots, sigs, threads):
    # Handle edge case where number of threads exceeds available data
    chunk_size = len(shots) // threads
    remainder = len(shots) % threads

    # If the number of threads does not evenly divide the array, ensure the chunks are sized properly
    if remainder != 0:
        shots = shots[:-(remainder)]  # Remove remainder to get equal-sized chunks

    # Reshape the integer array into chunks
    int_chunks = np.array(shots).reshape(threads, chunk_size)

    # Set up multiprocessing pool
    start = time.time()
    with Pool(threads) as pool:
        # Using starmap to send both the int_chunks and the full string array to the function
        results = pool.starmap(distributed_bench, [(chunk, sigs) for chunk in int_chunks])

    print(f'Distributed {len(shots)} shots {len(sigs)} signals {threads} threads {time.time()-start}) seconds')
    return time.time() - start

def HDF_BENCH(shots, sigs, threads):
    # Handle edge case where number of threads exceeds available data
    chunk_size = len(shots) // threads
    remainder = len(shots) % threads

    # If the number of threads does not evenly divide the array, ensure the chunks are sized properly
    if remainder != 0:
        shots = shots[:-(remainder)]  # Remove remainder to get equal-sized chunks

    # Reshape the integer array into chunks
    int_chunks = np.array(shots).reshape(threads, chunk_size)

    # Set up multiprocessing pool
    start = time.time()
    with Pool(threads) as pool:
        # Using starmap to send both the int_chunks and the full string array to the function
        results = pool.starmap(hdf_bench, [(chunk, sigs) for chunk in int_chunks])

    print(f'HDF {len(shots)} shots {len(sigs)} signals {threads} threads {time.time()-start}) seconds')
    return time.time() - start
# Run the benchmarks

import os
os.environ['default_tree_path'] = 'alcdata-archives::/cmod/trees/archives/~i~h/~g~f/~e~d/~t'
test_list = [1, 2, 4, 8, 16]    
thin_times = []
gm_times = []
distributed_times = []
hdf_times = []
c = MDSplus.Connection('alcdata-archives')
dummy = c.get('setenv("PyLib=python2.7")')
for threads in test_list:
    c.get('drop_caches()')
    hdf_times.append(HDF_BENCH(all_shots[0:500], signals[0:20], threads))
    c.get('drop_caches()')
    thin_times.append(THIN_BENCH(all_shots[0:500], signals[0:20], threads))
    c.get('drop_caches()')
    gm_times.append(GM_BENCH(all_shots[0:500], signals[0:20], threads))
    c.get('drop_caches()')
    distributed_times.append(DISTRIBUTED_BENCH(all_shots[0:500], signals[0:20], threads))

plt.plot(test_list, thin_times, label='Thin')
plt.plot(test_list, gm_times, label='GetMany')
plt.plot(test_list, distributed_times, label='Distributed')
plt.plot(test_list, hdf_times, label='HDF')
plt.yscale('log')
plt.xlabel('Number of Threads')
plt.ylabel('Time (s)')
plt.title('Benchmarking NO OS CACHING')
plt.legend()
plt.show()

