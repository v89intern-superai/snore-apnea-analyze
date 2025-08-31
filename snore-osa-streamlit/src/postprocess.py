import numpy as np
from typing import List, Dict

def hysteresis_series(p, enter=0.6, exit=0.4, min_consec=2):
    state, consec = 0, 0
    y = np.zeros_like(p, dtype=int)
    for i, v in enumerate(p):
        if state == 0 and v >= enter:
            consec += 1; 
            if consec >= min_consec: state, consec = 1, 0
        elif state == 1 and v <= exit:
            consec += 1;
            if consec >= min_consec: state, consec = 0, 0
        else:
            if state == 0 and v < enter: consec = 0
            if state == 1 and v > exit: consec = 0
        y[i] = state
    return y

def events_from_series(times, mask, min_ep_sec=10.0, merge_gap_sec=3.0, stride_s=1.0) -> List[Dict]:
    events = []; i = 0; n = len(mask)
    while i < n:
        if mask[i] == 1:
            start = times[i]; j = i+1
            while j < n and mask[j] == 1: j += 1
            end = times[j-1] + stride_s; dur = end - start
            if dur >= min_ep_sec:
                if events and start - events[-1]["end_s"] <= merge_gap_sec:
                    events[-1]["end_s"] = end
                    events[-1]["duration_s"] = events[-1]["end_s"] - events[-1]["start_s"]
                else:
                    events.append({"start_s": start, "end_s": end, "duration_s": dur})
            i = j
        else:
            i += 1
    return events
