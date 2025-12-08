# import os
# import time
# import psutil
import argparse
# import psutil
import argparse
# import threading
# import numpy as np
# PROC = psutil.Process(os.getpid())
# def _snap():
#     mem = PROC.memory_info().rss
#     cput = sum(PROC.cpu_times()[:2])  # user+system
#     return mem, cput

# def _fmt_mb(b):
#     return f"{b / (1024**2):.2f} MB"

# class PeakRSS:
#     def __init__(self, interval=0.05):
#         self.interval = interval
#         self.max_rss = 0
#         self._stop = threading.Event()
#         self._t = None
#     def _run(self):
#         while not self._stop.is_set():
#             rss = PROC.memory_info().rss
#             if rss > self.max_rss:
#                 self.max_rss = rss
#             time.sleep(self.interval)
#     def __enter__(self):
#         self.max_rss = PROC.memory_info().rss  # seed with current
#         self._t = threading.Thread(target=self._run, daemon=True)
#         self._t.start()
#         return self
#     def __exit__(self, exc_type, exc, tb):
#         self._stop.set()
#         self._t.join()


# # in ru.py: imports at file top: import psutil
import argparse, os, time, numpy as np

# # ru.py (replace/update existing CpuMonitor)
# import psutil
import argparse, os, time, threading, numpy as np

# class CpuMonitor:
#     """Continuously sample process-level CPU% (per-core percent). Minimal and robust for short tasks."""
#     def __init__(self, interval=0.01, include_children=False):
#         self.interval = interval
#         self.include_children = include_children
#         self.samples = []
#         self._stop = False

#     def __enter__(self):
#         self._proc = psutil.Process(os.getpid())
#         # prime so the next cpu_percent() returns a valid delta (first call returns 0)
#         self._proc.cpu_percent(interval=None)

#         self.samples = []
#         self._stop = False

#         def _run():
#             # tiny warmup so priming takes effect
#             time.sleep(self.interval)
#             while not self._stop:
#                 if self.include_children:
#                     # measure process + children (each call returns per-core percent for that proc)
#                     total = float(self._proc.cpu_percent(interval=None))
#                     for ch in self._proc.children(recursive=True):
#                         try:
#                             total += float(ch.cpu_percent(interval=None))
#                         except Exception:
#                             pass
#                     self.samples.append(total)
#                 else:
#                     # per-core percent for this process (float)
#                     self.samples.append(float(self._proc.cpu_percent(interval=None)))
#                 time.sleep(self.interval)

#         self._thread = threading.Thread(target=_run, daemon=True)
#         self._thread.start()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # allow one final sample to be taken
#         time.sleep(self.interval)
#         self._stop = True
#         self._thread.join()

#     def avg(self):
#         return float(np.mean(self.samples)) if self.samples else 0.0

#     def peak(self):
#         return float(np.max(self.samples)) if self.samples else 0.0

# import gc # <--- Add this import at the top of ru.py

# # --- Add these imports at the top of ru.py ---
# import os
# import time
# import psutil
import argparse
# import threading
# import numpy as np
# import gc  # <--- Make sure this is imported

# # ... (your other classes like PeakRSS, CpuMonitor go here) ...


# # --- Add this new, complete class ---

# class ResourceProfiler:
#     """
#     A context manager to profile wall time, peak MARGINAL RAM,
#     and CPU utilization (avg/peak) for a block of code.
    
#     Measures CPU for parent process + all children.
#     Measures RAM added *by the code block*, not total process RAM.
#     """
#     def __init__(self, interval=0.01):
#         self.interval = interval
#         self.proc = psutil.Process(os.getpid())
#         self.n_cores = psutil.cpu_count(logical=True) or 1
        
#         # Results will be stored here
#         self.wall_seconds = 0
#         self.cpu_avg_machine_pct = 0
#         self.cpu_peak_machine_pct = 0
#         self.peak_ram_mb = 0
#         self.baseline_mem = 0

#     def __enter__(self):
#         # 1. Force garbage collection and get baseline memory
#         gc.collect()
#         self.baseline_mem = self.proc.memory_info().rss

#         # 2. Start monitors
#         # We pass include_children=True to fix the CPU bug
#         self.rss_mon = PeakRSS(self.interval)
#         self.cpu_mon = CpuMonitor(self.interval, include_children=True) 
        
#         # We must re-seed the rss_mon's max with the baseline
#         self.rss_mon.max_rss = self.baseline_mem
        
#         # Start the monitors
#         self.rss_mon.__enter__()
#         self.cpu_mon.__enter__()
        
#         # 3. Start timers
#         self.t0 = time.perf_counter()
#         self.cpu_time0 = self.proc.cpu_times().user + self.proc.cpu_times().system
#         try:
#             # Add time from all children
#             self.cpu_time0 += sum((ch.cpu_times().user + ch.cpu_times().system) for ch in self.proc.children(recursive=True))
#         except psutil.NoSuchProcess:
#              pass # Children can die quickly
        
#         return self # This is the object you get in 'as profiler'

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # 1. Stop timers
#         self.t1 = time.perf_counter()
#         self.cpu_time1 = self.proc.cpu_times().user + self.proc.cpu_times().system
#         try:
#             # Add time from all children
#             self.cpu_time1 += sum((ch.cpu_times().user + ch.cpu_times().system) for ch in self.proc.children(recursive=True))
#         except psutil.NoSuchProcess:
#             pass # Children can die quickly

#         # 2. Stop monitors
#         self.rss_mon.__exit__(None, None, None)
#         self.cpu_mon.__exit__(None, None, None)
        
#         # 3. Calculate and store results
        
#         # --- Wall Time ---
#         self.wall_seconds = max(1e-12, self.t1 - self.t0)
        
#         # --- CPU ---
#         proc_cpu_seconds = max(0.0, self.cpu_time1 - self.cpu_time0)
        
#         # Avg CPU (deterministic, 0-100% of machine)
#         self.cpu_avg_machine_pct = (proc_cpu_seconds / (self.wall_seconds * self.n_cores)) * 100.0
        
#         # Peak CPU (sampled, 0-100% of machine)
#         peak_cpu_percent = self.cpu_mon.peak() if getattr(self.cpu_mon, "samples", None) else 0.0
#         self.cpu_peak_machine_pct = (peak_cpu_percent / self.n_cores)

#         # Clamp and round
#         self.cpu_avg_machine_pct = round(max(0.0, min(100.0, self.cpu_avg_machine_pct)), 3)
#         self.cpu_peak_machine_pct = round(max(0.0, min(100.0, self.cpu_peak_machine_pct)), 3)
        
#         # --- Peak RAM (MODIFIED) ---
#         # Calculate the *difference* from the baseline
#         marginal_peak_bytes = max(0, self.rss_mon.max_rss - self.baseline_mem)
#         self.peak_ram_mb = round(marginal_peak_bytes / (1024**2), 2)



import os
import time
import psutil
import argparse
import threading
import numpy as np
import gc  # gc is still needed for the helper classes, but not in the profiler

# Store the main process
PROC = psutil.Process(os.getpid())

# --- Helper Class 1: PeakRSS (Monitors TOTAL RAM) ---
class PeakRSS:
    """Monitors peak RSS memory in a background thread."""
    def __init__(self, interval=0.05):
        self.interval = interval
        self.max_rss = 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = PROC.memory_info().rss
                if rss > self.max_rss:
                    self.max_rss = rss
            except psutil.NoSuchProcess:
                break # Process ended
            time.sleep(self.interval)

    def __enter__(self):
        try:
            self.max_rss = PROC.memory_info().rss  # seed with current
        except psutil.NoSuchProcess:
            self.max_rss = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._t.join()

# --- Helper Class 2: CpuMonitor (Includes Children) ---
class CpuMonitor:
    """Continuously sample process-level CPU% (per-core percent)."""
    def __init__(self, interval=0.01, include_children=False):
        self.interval = interval
        self.include_children = include_children
        self.samples = []
        self._stop = False
        self._thread = None
        self._proc = psutil.Process(os.getpid())

    def __enter__(self):
        try:
            self._proc.cpu_percent(interval=None)
            if self.include_children:
                for ch in self._proc.children(recursive=True):
                    ch.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            pass 

        self.samples = []
        self._stop = False

        def _run():
            time.sleep(self.interval) 
            while not self._stop:
                try:
                    total = float(self._proc.cpu_percent(interval=None))
                    if self.include_children:
                        for ch in self._proc.children(recursive=True):
                            try:
                                total += float(ch.cpu_percent(interval=None))
                            except psutil.NoSuchProcess:
                                pass # Child died
                    self.samples.append(total)
                except psutil.NoSuchProcess:
                    break # Main process died
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        time.sleep(self.interval) 
        self._stop = True
        self._thread.join()

    def avg(self):
        return float(np.mean(self.samples)) if self.samples else 0.0

    def peak(self):
        return float(np.max(self.samples)) if self.samples else 0.0

# --- Main Profiler Class (Goal 2: Total RAM + Nanoseconds) ---
class ResourceProfiler:
    """
    A context manager to profile wall time, PEAK TOTAL RAM,
    and CPU utilization (avg/peak) for a block of code.
    
    Measures CPU for parent process + all children.
    Measures TOTAL process RAM (high-water mark).
    Uses high-precision nanosecond timers.
    """
    def __init__(self, interval=0.01):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.n_cores = psutil.cpu_count(logical=True) or 1
        
        # Results will be stored here
        self.wall_nanoseconds = 0
        self.wall_seconds = 0.0
        self.cpu_avg_machine_pct = 0.0
        self.cpu_peak_machine_pct = 0.0
        self.peak_ram_mb = 0.0

    def __enter__(self):
        # 1. Start monitors
        # We pass include_children=True to fix the CPU bug
        self.rss_mon = PeakRSS(self.interval)
        self.cpu_mon = CpuMonitor(self.interval, include_children=True) 
        
        # Start the monitors
        self.rss_mon.__enter__()
        self.cpu_mon.__enter__()
        
        # 2. Start timers
        self.t0_ns = time.perf_counter_ns()
        try:
            self.cpu_time0 = self.proc.cpu_times().user + self.proc.cpu_times().system
            # Add time from all children
            self.cpu_time0 += sum((ch.cpu_times().user + ch.cpu_times().system) for ch in self.proc.children(recursive=True))
        except psutil.NoSuchProcess:
             self.cpu_time0 = 0 # Process died right at the start
        
        return self # This is the object you get in 'as profiler'

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Stop timers
        self.t1_ns = time.perf_counter_ns()
        try:
            self.cpu_time1 = self.proc.cpu_times().user + self.proc.cpu_times().system
            # Add time from all children
            self.cpu_time1 += sum((ch.cpu_times().user + ch.cpu_times().system) for ch in self.proc.children(recursive=True))
        except psutil.NoSuchProcess:
            self.cpu_time1 = self.cpu_time0 # Process died, no time elapsed

        # 2. Stop monitors
        self.rss_mon.__exit__(None, None, None)
        self.cpu_mon.__exit__(None, None, None)
        
        # 3. Calculate and store results
        
        # --- Wall Time ---
        # Calculate total nanoseconds first
        self.wall_nanoseconds = max(1, self.t1_ns - self.t0_ns) 
        # Derive float seconds from nanoseconds for CPU avg calculation
        self.wall_seconds = self.wall_nanoseconds / 1_000_000_000.0
        
        # --- CPU ---
        proc_cpu_seconds = max(0.0, self.cpu_time1 - self.cpu_time0)
        
        # Avg CPU (deterministic, 0-100% of machine)
        self.cpu_avg_machine_pct = (proc_cpu_seconds / (self.wall_seconds * self.n_cores)) * 100.0
        
        # Peak CPU (sampled, 0-100% of machine)
        peak_cpu_percent = self.cpu_mon.peak() if getattr(self.cpu_mon, "samples", None) else 0.0
        self.cpu_peak_machine_pct = (peak_cpu_percent / self.n_cores)

        # Clamp and round
        self.cpu_avg_machine_pct = round(max(0.0, min(100.0, self.cpu_avg_machine_pct)), 3)
        self.cpu_peak_machine_pct = round(max(0.0, min(100.0, self.cpu_peak_machine_pct)), 3)
        
        # --- Peak RAM (TOTAL) ---
        # Calculate TOTAL process peak RAM
        self.peak_ram_mb = round(self.rss_mon.max_rss / (1024**2), 2)