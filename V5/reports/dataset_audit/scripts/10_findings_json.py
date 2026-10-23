"""
Assemble audit_findings.json: measured summaries (from _work/*.json) plus the
curated, evidence-classified findings that the Markdown reports discuss.

evidence classes:
  verified_file      - verified from file contents (script named in 'source')
  verified_local_doc - verified from local documentation
  external           - supported by external publication / official page (URL)
  inferred           - inferred, reasoning in 'note'
  unknown            - requires further investigation

Usage:  py -P -E 10_findings_json.py <audit_dir>
"""

import json
import os
import sys


def F(claim, ev, source, note=""):
    d = {"claim": claim, "evidence": ev, "source": source}
    if note:
        d["note"] = note
    return d


URL = {
    "wicount_paper": "https://arxiv.org/abs/2412.06861",
    "m80": "https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels",
    "sharp": "https://github.com/francescamen/SHARP/blob/main/Python_code/CSI_phase_sanitization_signal_preprocessing.py",
    "countfi": "https://guojingtao-1997.github.io/CountFi/",
    "countfi_paper": "https://arxiv.org/abs/2410.11400",
    "wimans": "https://github.com/huangshk/WiMANS",
    "wimans_paper": "https://arxiv.org/abs/2402.09430",
    "iitbhu": "https://ieee-dataport.org/documents/channel-state-information-dataset-multi-human-activity-recognition-indoor-environments",
    "csihar": "https://doi.org/10.6084/m9.figshare.14386892",
    "csihar_paper": "https://doi.org/10.1016/j.procs.2021.12.211",
    "haron": "https://huggingface.co/datasets/Haron98/WIFI-CSI-Dataset",
    "espidf": "https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi-driver/wifi-vendor-features.html",
    "esp32csitool": "https://github.com/StevenMHernandez/ESP32-CSI-Tool",
}


def main():
    audit = sys.argv[1]
    work = os.path.join(audit, "_work")
    J = lambda n: json.load(open(os.path.join(work, n), encoding="utf-8"))
    inv = {r["dataset_dir"]: r for r in J("inventory.json")["datasets"]}
    wc, m80, cf, wm, ib, ibm, ch, hr = (J("wicount.json"), J("mat80.json"), J("countfi.json"),
                                        J("wiman.json"), J("iitbhu.json"), J("iitbhu_macs.json"),
                                        J("csihar.json"), J("haron.json"))

    def disk(name):
        r = inv[name]
        return {"files": r["n_files"], "bytes": r["total_bytes"], "gib": r["total_gib"],
                "extensions": r["extensions"], "archives": len(r["archives"])}

    datasets = [
        {
            "dataset": "WiCount", "path": "D:/wifi/Datasets/WiCount", "audit_status": "fully_audited",
            "disk": disk("WiCount"),
            "measured": {"by_label": wc["by_label"], "csi_value_range": wc["csi"]["value_range"],
                         "always_zero_positions": wc["csi"]["always_zero_positions"]},
            "findings": [
                F("15 CSV recordings, 60 s each; classes 0:1, 1:7, 2:6, 3:1 files", "verified_file", "scripts/02_wicount.py"),
                F("Folder name = number of people (0-3)", "verified_local_doc", "WiCount/README.md"),
                F("ESP32-S3 receiver, 1 antenna, 2.4 GHz, 20 MHz, 52 subcarriers, ~100 Hz nominal", "external", URL["wicount_paper"]),
                F("Observed packet spacing median 11.99 ms (~83 Hz), mean rate 80-94 Hz", "verified_file", "scripts/02_wicount.py"),
                F("All frames legacy non-HT (sig_mode=0, MCS 0), channel 12; 22 of 28 columns constant", "verified_file", "scripts/02_wicount.py"),
                F("Single TX MAC / single receiver stream although README mentions multiple receivers", "verified_file", "scripts/02_wicount.py"),
                F("CSI has 104 ints with no always-zero position: null tones already removed; tone indices undocumented", "verified_file", "scripts/02_wicount.py"),
                F("Espressif stores each CSI item as imaginary then real; README states real then imaginary", "external", URL["espidf"]),
                F("process1.py builds imag part from x[2i-1] (should be x[2i+1]) - indexing bug", "verified_local_doc", "WiCount/data_process_example/process1.py"),
                F("Recordings made in order 0->1->2->3 people within 45 min of one evening: count confounded with time", "verified_file", "filenames/timestamps"),
                F("Paper says TX is a home router; README says ESP32-S3 TX", "external", URL["wicount_paper"], "conflict between primary sources"),
            ],
            "uncertainties": ["which 52 of 64 tones are kept", "TX hardware", "room geometry"],
        },
        {
            "dataset": "80MHz Human Sensing - PC folders", "path": "D:/wifi/Datasets/Dataset",
            "audit_status": "fully_audited_structure",
            "disk": disk("Dataset"),
            "measured": {k: {kk: v.get(kk) for kk in ("shape", "complex", "real_stored_type", "real_range",
                                                      "low_energy_bins", "saturated_fraction",
                                                      "lag_corr_amplitude", "rows_divisible_by_4")}
                         for k, v in m80.items()},
            "findings": [
                F("Each MAT holds one variable csi_buff (N x 256), complex, stored int16, range +-2047", "verified_file", "scripts/07_mat80.py"),
                F("Rows interleave 4 RX chains: lag-4 amplitude correlation 0.82-0.9998 vs lag-1 0.05-0.44", "verified_file", "scripts/07_mat80.py"),
                F("SHARP code reads rows with stride n_ss*n_core and deletes 14 bins after fftshift; negates upper columns", "external", URL["sharp"]),
                F("Low-energy bins 1,123-133,255 in natural FFT order (guards + DC) -> 242 usable tones", "verified_file", "scripts/07_mat80.py"),
                F("Filename PCxy_nN: N = concurrent people; 802.11ac ch42 80 MHz; 173 pkt/s; monitor RT-AC86U 4 antennas", "external", URL["m80"]),
                F("n01 absent in all four PC setups", "verified_file", "inventory"),
                F("PC2a_n00.mat contains a single all-zero 1x256 uint8 row", "verified_file", "scripts/07_mat80.py"),
                F("PC1a_n05 has 178,197 rows (not divisible by 4)", "verified_file", "scripts/07_mat80.py"),
                F("No timestamps, RSSI or packet metadata stored", "verified_file", "scripts/07_mat80.py"),
            ],
            "uncertainties": ["meaning of setups PC-1a..PC-4a (set_names.pdf not local)", "monitor chipset not named on page",
                              "people behaviour in PC traces (secondary source: moving)"],
        },
        {
            "dataset": "CountFi (local Apr-2025 subset)", "path": "D:/wifi/Datasets/CountFi",
            "audit_status": "fully_audited_signal_labels_unverified",
            "disk": disk("CountFi"),
            "measured": {"aggregate": cf["aggregate"], "csi_sample": cf["csi_sample"]},
            "findings": [
                F("4,740 pcaps x 1,000 Nexmon-CSI UDP packets; 10 empty and 4 header-only files", "verified_file", "scripts/08_countfi.py"),
                F("chanspec 0xe02a (5 GHz ch42 80 MHz), core/ss 0, single source MAC in 100% of packets", "verified_file", "scripts/08_countfi.py"),
                F("CSI is int16 (real, imag) pairs: smooth VHT80 spectrum on QoS frames; packed-float decode yields noise", "verified_file", "scripts/08b_countfi_decode_test.py"),
                F("53% of packets are Block-Acks whose CSI has non-HT-duplicate 4x20 MHz structure", "verified_file", "scripts/08b_countfi_decode_test.py", "frame-type interpretation inferred from 802.11ac duplicate format"),
                F("Capture rate ~1 kHz; all captures on 2025-04-08 06:37-09:43 UTC", "verified_file", "scripts/08_countfi.py"),
                F("head/end folders are simultaneous receivers (start times within ~20 ms)", "verified_file", "scripts/08_countfi.py"),
                F("Apr-2025 set = mixed moving/stationary; _xm suffix where x = section; 1,000 samples per pcap; ground truth manually recorded", "external", URL["countfi"]),
                F("Numbered subfolders are contiguous time blocks that change along the trip; interpreted as passenger count", "inferred", "scripts/08_countfi.py",
                  "not documented; ground-truth log absent"),
                F("Licence CC BY-NC 4.0", "external", URL["countfi"]),
                F("Full decode of all 4.74M packets: 0 truncated/invalid packets, 0 all-zero CSI, 0 repeated payloads, no int16 saturation", "verified_file", "scripts/08c_countfi_full_decode.py"),
                F("Bins 0-1 carry junk values in every packet; bins 125-131 constant over time in both receivers; standard 14-null removal leaves 242 clean tones", "verified_file", "scripts/08c_countfi_full_decode.py"),
                F("head vs end differ only in constant guard-bin values and receiver gain profile", "verified_file", "scripts/08c_countfi_full_decode.py"),
            ],
            "uncertainties": ["subfolder semantics", "condition codes ms, mm_v2, mm_v3, and '48'", "chip id 0x0065 model",
                              "labels only - signal fully decoded"],
        },
        {
            "dataset": "WiMANS (amplitude subset)", "path": "D:/wifi/Datasets/Wiman",
            "audit_status": "partially_audited_labels_absent",
            "disk": disk("Wiman"),
            "measured": {k: wm[k] for k in ("n_files", "header_dtypes", "first_id_range", "n_first_ids",
                                            "files_per_first_id", "header_errors")},
            "findings": [
                F("3,057 float32 arrays of shape (T,3,3,30), T 2,772-2,986; headers read for all files", "verified_file", "scripts/06_wiman.py"),
                F("Axis order time x TX x RX x subcarrier; 1 kHz, 3 s; Intel 5300; 11,286 samples; users 0-5", "external", URL["wimans_paper"]),
                F("Max amplitude 179.605 = sqrt(127^2+127^2): unnormalised magnitude of 8-bit I/Q", "verified_file", "scripts/06_wiman.py"),
                F("annotation.csv, raw .mat and video are not present: no labels locally", "verified_file", "inventory"),
                F("act_24_89.npy failed earlier validation and is now absent; act_24 has 87 files", "verified_file", "Wiman/validation_summary.txt + listing"),
                F("Lag-1 temporal autocorrelation ~0 for groups 1,10-18,201-206 vs 0.37-0.95 for 19-24,100-108", "verified_file", "scripts/06_wiman.py", "cause unknown"),
            ],
            "uncertainties": ["labels", "cause of low temporal autocorrelation", "which band/environment each group is"],
        },
        {
            "dataset": "IIT-BHU multi-human activity", "path": "D:/wifi/Datasets/IIT-BHU",
            "audit_status": "fully_audited_files_no_local_docs",
            "disk": disk("IIT-BHU"),
            "measured": {k: ib[k] for k in ("csi_len", "rows_per_file", "duration_s", "rate_hz", "n_users_by_group",
                                            "user_overlap_between_groups", "bad_lines")} | {"mac_mix": ibm["summary"]},
            "findings": [
                F("ESP32-CSI-Tool CSI_RAW line format with host epoch appended", "verified_file", "scripts/04_iitbhu.py"),
                F("Format of the tool's CSI_DATA line", "external", URL["esp32csitool"]),
                F("384-value rows (HT40 LLTF+HT-LTF) 96.8%, 128-value 3.2%, ~40 corrupt lengths, 2,452 garbled lines", "verified_file", "scripts/04_iitbhu.py"),
                F("Packet rate per file 4.1-51.6 Hz (median 27.7)", "verified_file", "scripts/04_iitbhu.py"),
                F("Dominant TX MAC median 98.5% of rows; 3 different dominant TX MACs across files", "verified_file", "scripts/04b_iitbhu_macs.py"),
                F("11 byte-identical file pairs carry different user-ID filenames; 3 empty files", "verified_file", "scripts/01_inventory.py"),
                F("Dataset: ESP32 NodeMCU, ESP32-CSI Toolkit, 80+ participants, 6 activities, 1/2/3-user scenarios", "external", URL["iitbhu"]),
                F("No 0-person recordings", "verified_file", "listing"),
            ],
            "uncertainties": ["room layout", "how activity order maps to users in multi-user files", "licence terms beyond DataPort"],
        },
        {
            "dataset": "CSI HAR (Zhuravchak & Kapshii)", "path": "D:/wifi/Datasets/CSI HAR",
            "audit_status": "fully_audited_files",
            "disk": disk("CSI HAR"),
            "measured": {"sessions": [{k: s[k] for k in ("session", "data_rows", "label_rows", "label_boxes_rows",
                                                          "label_distribution")} for s in ch["sessions"]]},
            "findings": [
                F("Rows = 114 constant index ints + 4x114 amplitudes + 4x114 phases", "verified_file", "scripts/05_csihar.py"),
                F("Amplitude/phase are exact polar form of integer I/Q (|I|,|Q| <= ~404)", "verified_file", "scripts/05_csihar.py"),
                F("TP-Link TL-WDR4300, Atheros CSI Tool, 5 GHz ch60, 40 MHz, 2x2, 114 subcarriers, 1 person, 3 rooms", "external", URL["csihar"]),
                F("Each CSI packet labelled with an activity and a camera bounding box", "external", URL["csihar_paper"]),
                F("label.csv has one row fewer than data.csv in every session; label_boxes differs by 0-8 rows; room_2 lacks boxes", "verified_file", "scripts/05_csihar.py"),
                F("No timestamps or RSSI: sampling rate cannot be determined", "verified_file", "scripts/05_csihar.py"),
            ],
            "uncertainties": ["packet rate", "antenna-pair order of the 4 blocks", "exact label alignment"],
        },
        {
            "dataset": "Haron98 WiFi CSI Router Reception", "path": "D:/wifi/Datasets/Haron98_WIFI-CSI-Dataset",
            "audit_status": "fully_audited_files_label_semantics_unknown",
            "disk": disk("Haron98_WIFI-CSI-Dataset"),
            "measured": {k: hr[k] for k in ("lines_total", "lines_valid", "frame_interval_s", "trial_duration_s",
                                            "time_range", "dev1_amplitude_by_label", "rssi_by_label")},
            "findings": [
                F("4,800 files x 100 lines; 479,402 valid; malformed always line 1", "verified_file", "scripts/03_haron.py"),
                F("ESP32 LLTF 64 tones; zero pairs 0 and 27-37 match ESP-IDF ordering", "verified_file", "scripts/03_haron.py"),
                F("Passive router traffic: HT, MCS 7, ch 6, 20 MHz; ~25 Hz", "verified_file", "scripts/03_haron.py"),
                F("label_00..03 meanings are TODO; licence contradictory", "verified_local_doc", "README.md"),
                F("label_00 recorded last (22:44-23:21) for all subjects; higher mean amplitude, lower temporal variance",
                  "verified_file", "scripts/03_haron.py", "consistent with an empty/static baseline but confounded with time - not a label definition"),
            ],
            "uncertainties": ["label meanings", "ESP32 model", "room"],
        },
    ]

    out = {
        "audit_date": "2026-10-19",
        "datasets_root": "D:/wifi/Datasets",
        "scope_note": "EHUNAM is referenced in D:/wifi notebooks but not present locally; excluded.",
        "environment": {"disk_free_gib_D": 8.3, "ram_total_gib": 15.6, "ram_free_gib_at_start": 2.0,
                        "python": "3.12.10", "added_packages": ["h5py 3.16.0", "dpkt 1.9.8"]},
        "summary": {
            "datasets_discovered": 7,
            "fully_audited": ["WiCount", "80MHz PC", "IIT-BHU", "CSI HAR", "Haron98", "CountFi (signal; labels unverified)"],
            "partially_audited": ["WiMANS"],
            "removed_from_disk_2026_10_19": ["WiMANS (Wiman/)", "Haron98"],
            "blocked": [],
            "direct_counting_candidates": ["80MHz PC (0,2-10)", "WiCount (0-3, ESP32-S3)",
                                           "CountFi (conditional on label verification)",
                                           "WiMANS (only after annotation.csv)"],
            "related_task_only": ["IIT-BHU (activity; 1-3 user scenarios)", "CSI HAR (presence/HAR)",
                                  "Haron98 (undefined labels)"],
        },
        "datasets": datasets,
        "resources": {
            "note": "RAM figures are estimates (labelled), disk figures measured",
            "per_dataset_ram_estimate_mb": {
                "WiCount": "<50 (whole dataset)",
                "80MHz PC": "largest file as complex128 ~850; complex64 ~425; streaming reader <150 (measured run)",
                "CountFi": "~1 per pcap; whole set as int16 ~4,850",
                "WiMANS": "~3.3 per sample; whole local set ~9,000",
                "IIT-BHU": "whole set as int16 ~750",
                "CSI HAR": "largest session float64 ~107; all ~555",
                "Haron98": "whole set int16 ~125",
            },
            "temporary_disk_needed": "none for audit (no archives); conversions should write float16/int16 to keep < 5 GB",
            "cloud_recommended_for": ["full WiMANS download incl. annotation/raw (~tens of GB > 8.3 GiB free)",
                                      "any full 80MHz or CountFi tensor materialisation in float32"],
        },
    }
    path = os.path.join(audit, "audit_findings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(path, os.path.getsize(path))


if __name__ == "__main__":
    main()
