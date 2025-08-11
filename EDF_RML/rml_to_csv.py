# rml_to_csv.py
from __future__ import annotations
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import csv
import argparse

# ---------------- XML parsing utilities ----------------

def _clean_xml(xml_text: str) -> str:
    """Remove namespaces and tag prefixes so we can parse without namespace maps."""
    xml_text = re.sub(r'xmlns(:\w+)?="[^"]+"', '', xml_text)
    xml_text = re.sub(r'<(/?)(\w+):', r'<\1', xml_text)
    return xml_text

def parse_single_rml(rml_path: Path) -> Dict:
    """
    Parse one PSG-Audio RML file.
    Returns:
        {
          "RecordingStart": str|None (ISO),
          "Segments": [{"StartTime": str|None, "Duration": float|None}, ...],
          "Events": [{"Family": str, "Type": str, "Start": float, "Duration": float}, ...]
        }
    """
    text = rml_path.read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(_clean_xml(text))

    rec_start = (root.find(".//RecordingStart").text
                 if root.find(".//RecordingStart") is not None else None)

    segments = []
    for seg in root.findall(".//Sessions/Session/Segments/Segment"):
        st = seg.findtext("StartTime")
        dur_txt = seg.findtext("Duration")
        dur = float(dur_txt) if dur_txt else None
        segments.append({"StartTime": st, "Duration": dur})

    events = []
    # Primary PSG-Audio format: <Events><Event Family="..." Type="..." Start="..." Duration="..."/>
    for ev in root.findall(".//Event"):
        fam = ev.attrib.get("Family")
        typ = ev.attrib.get("Type")
        st = ev.attrib.get("Start")
        dur = ev.attrib.get("Duration")
        if st is None or dur is None:
            continue
        events.append({
            "Family": fam, "Type": typ,
            "Start": float(st), "Duration": float(dur)
        })

    # Optional fallback for ScoredEvent-style (not typical for PSG-Audio)
    for ev in root.findall(".//ScoredEvent"):
        concept = ev.findtext("EventConcept")
        st = ev.findtext("Start")
        dur = ev.findtext("Duration")
        fam = ev.findtext("Family")
        if st and dur:
            events.append({
                "Family": fam, "Type": concept,
                "Start": float(st), "Duration": float(dur)
            })

    return {"RecordingStart": rec_start, "Segments": segments, "Events": events}

# ---------------- Merge utilities ----------------

def cumulative_segments(segments: List[Dict]) -> List[Tuple[float, float]]:
    """
    Build cumulative [t0, t1) windows over all segments using their Durations.
    Returns list of (t0, t1) in seconds from global start.
    """
    out, t = [], 0.0
    for seg in segments:
        dur = float(seg["Duration"] or 0.0)
        out.append((t, t + dur))
        t += dur
    return out

def locate_segment(t_global: float, cum_segs: List[Tuple[float, float]]) -> int:
    """Return segment index where t_global falls, or -1 if not found."""
    for i, (t0, t1) in enumerate(cum_segs):
        if t0 <= t_global < t1:
            return i
    return -1

def merge_events_from_files(rml_files: List[Path]) -> Dict:
    """
    Merge multiple RML files of the same patient.
    Deduplicate events by (Type, Start, Duration) triple.
    Returns dict with combined RecordingStart (first non-empty),
    combined Segments (from the file that has them), and merged Events.
    """
    recording_start = None
    segments = []
    merged = {}
    for p in rml_files:
        info = parse_single_rml(p)
        if recording_start is None and info.get("RecordingStart"):
            recording_start = info["RecordingStart"]
        if not segments and info.get("Segments"):
            segments = info["Segments"]  # choose the first complete set

        for e in info.get("Events", []):
            key = (e.get("Type"), e.get("Start"), e.get("Duration"))
            if key not in merged:
                merged[key] = e

    events = list(merged.values())
    return {"RecordingStart": recording_start, "Segments": segments, "Events": events}

# ---------------- CSV export ----------------

def export_patient_csv(out_csv: Path, patient_id: str, rml_files: List[Path]) -> None:
    """
    Export a single patient's merged events into a CSV.
    Columns:
        patient_id, event_id, family, type, start_sec, duration_sec, end_sec,
        segment_index, segment_local_start_sec, recording_start_iso
    """
    merged = merge_events_from_files(rml_files)
    events = merged["Events"]
    segments = merged["Segments"] or []
    cum_segs = cumulative_segments(segments) if segments else []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "patient_id", "event_id", "family", "type",
            "start_sec", "duration_sec", "end_sec",
            "segment_index", "segment_local_start_sec",
            "recording_start_iso"
        ])

        for idx, e in enumerate(sorted(events, key=lambda x: x["Start"])):
            start = float(e["Start"])
            dur = float(e["Duration"])
            end = start + dur
            seg_idx = locate_segment(start, cum_segs) if cum_segs else -1
            seg_local = start - cum_segs[seg_idx][0] if seg_idx >= 0 else None
            w.writerow([
                patient_id, idx, e.get("Family"), e.get("Type"),
                f"{start:.3f}", f"{dur:.3f}", f"{end:.3f}",
                seg_idx, f"{seg_local:.3f}" if seg_local is not None else "",
                merged.get("RecordingStart") or ""
            ])

# ---------------- Batch runner ----------------

def infer_patient_id(name: str) -> str:
    """
    Infer patient id from filename pattern like 00000995-100507.rml or 00000995-..._1.rml.
    Returns the leading numeric block as string, or the stem if not matched.
    """
    m = re.match(r"^(\d{6,})", name.replace(" ", ""))
    return m.group(1) if m else Path(name).stem

def collect_patient_groups(folder: Path) -> Dict[str, List[Path]]:
    """
    Group RML files by patient id. Accepts variations like:
    00000995-100507.rml, 00000995-100507_1.rml, 00000995-100507_2.rml
    """
    groups: Dict[str, List[Path]] = {}
    for p in folder.glob("*.rml"):
        pid = infer_patient_id(p.name)
        groups.setdefault(pid, []).append(p)
    return groups

def main():
    parser = argparse.ArgumentParser(description="Convert PSG-Audio RML to per-patient CSV.")
    parser.add_argument("--input", required=True, help="Folder containing RML files.")
    parser.add_argument("--output", required=True, help="Folder to write patient CSVs.")
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Optional list of patient ids to export (e.g. 00000995 00000999).")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    groups = collect_patient_groups(in_dir)

    # Narrow to selected patients if provided
    if args.patients:
        groups = {pid: files for pid, files in groups.items() if pid in args.patients}

    if not groups:
        print("No RML files found or no matching patients.")
        return

    print(f"Found {len(groups)} patient(s). Exporting CSVs...")
    for pid, files in sorted(groups.items()):
        out_csv = out_dir / f"{pid}.csv"
        try:
            export_patient_csv(out_csv, pid, sorted(files))
            print(f"Wrote {out_csv}")
        except Exception as e:
            print(f"Failed {pid}: {e}")

if __name__ == "__main__":
    main()
