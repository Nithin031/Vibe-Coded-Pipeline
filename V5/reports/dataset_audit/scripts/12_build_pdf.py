"""
Build the summary PDF: audit findings + usability + step-by-step plan.

Output: D:/wifi/reports/CSI_People_Counting_Audit_and_Plan.pdf
Text is kept to WinAnsi characters (Helvetica) so nothing renders as boxes.

Usage:  py -P -E 12_build_pdf.py <out_pdf>
"""

import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


# ----------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------

BASE = getSampleStyleSheet()
INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
SOFT = colors.HexColor("#e8eef6")
WARN = colors.HexColor("#fdecea")
OK = colors.HexColor("#e7f5ec")
MID = colors.HexColor("#fff7e0")

H1 = ParagraphStyle("H1", parent=BASE["Heading1"], fontSize=17, leading=21, textColor=ACCENT,
                    spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=BASE["Heading2"], fontSize=13, leading=16, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=BASE["Heading3"], fontSize=11, leading=14, textColor=INK,
                    spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=BASE["BodyText"], fontSize=9.5, leading=13, textColor=INK,
                      alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8, leading=10.2, spaceAfter=0)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("Title", parent=BASE["Title"], fontSize=22, leading=27, textColor=ACCENT,
                       alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("Sub", parent=BODY, fontSize=11, leading=15, textColor=colors.HexColor("#52606d"))


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BUL):
    return [Paragraph(t, style, bulletText="-") for t in items]


def checklist(items):
    return [Paragraph(t, BUL, bulletText="[ ]") for t in items]


def table(rows, widths, header=True, shade=None):
    data = [[P(c, CELLB if (header and r == 0) else CELL) if isinstance(c, str) else c
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcccdc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    for (r, c) in (shade or {}).items():
        st.append(("BACKGROUND", (0, r), (-1, r), c))
    t.setStyle(TableStyle(st))
    return t


def box(paragraphs, bg=SOFT):
    t = Table([[paragraphs]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#7b8794"))
    canvas.drawString(18 * mm, 10 * mm, "Wi-Fi CSI people counting - dataset audit & plan - 2026-10-19")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------

def build(out):
    s = []

    # ---------------- Title + bottom line ----------------
    s.append(P("Wi-Fi CSI People Counting", TITLE))
    s.append(P("Dataset audit findings, what is usable, and a step-by-step plan to an ESP32-S3 people counter", SUB))
    s.append(Spacer(1, 6))
    s.append(P("Prepared 2026-10-19 from the audit in D:/wifi/reports/dataset_audit/. "
               "Every claim below was checked against the files, the dataset's own documentation, "
               "or the official dataset page / paper. Items not yet verified are marked <b>(unverified)</b>.", SMALL))
    s.append(Spacer(1, 8))

    s.append(box([
        P("<b>Bottom line</b>", H3),
        *bullets([
            "<b>Datasets with real people-count labels:</b> EHUNAM (0-8), WiMANS (0-5), the 80 MHz set in "
            "<i>Dataset/</i> (0 and 2-10), CountFi (1-8 and 10, very likely) and WiCount (0-3).",
            "<b>Only WiCount was recorded on an ESP32-S3</b>, and it has just 15 one-minute files covering 0-3 people. "
            "No public dataset combines ESP32-S3 hardware with a wide count range.",
            "<b>So the plan is:</b> learn what makes counting work from the large labelled sets, pre-train on "
            "CSI-Bench and the ESP32 sets, then <b>record your own ESP32-S3 data</b>. That last step decides "
            "real-world accuracy.",
            "<b>Related tasks only:</b> IIT-BHU (1-3 person activity scenarios), CSI HAR (empty vs one person), "
            "Haron98 (labels undefined) and CSI-Bench (no count labels at all).",
        ]),
    ]))

    # ---------------- 1. What was done ----------------
    s.append(P("1. What was audited", H1))
    s.append(P("Seven folders in D:/wifi/Datasets were inspected file by file, read-only. Nothing was trained, "
               "resampled or normalised. For each dataset the audit established: what the files contain, "
               "every field and its meaning, the hardware and radio settings, how labels were assigned, "
               "signal quality and timing, and leakage risks. EHUNAM, the full WiMANS and CSI-Bench live in "
               "your Google Colab and have <b>not</b> been audited yet. Their entries below come from their papers "
               "or your survey PDF."))
    s.append(table([
        ["Dataset", "Where", "Audit depth"],
        ["WiCount", "D:/wifi/Datasets/WiCount", "Every row of all 15 files"],
        ["80 MHz Human Sensing (Dataset/)", "D:/wifi/Datasets/Dataset", "Every row of all 40 MAT files (streamed)"],
        ["CountFi (Apr 2025)", "D:/wifi/Datasets/CountFi", "All 4.74 million packets decoded"],
        ["IIT-BHU", "D:/wifi/Datasets/IIT-BHU", "Every line of all 928 files"],
        ["CSI HAR", "D:/wifi/Datasets/CSI HAR", "Every row of all 10 sessions"],
        ["Haron98", "Colab (deleted locally)", "Every line of all 4,800 files, plus one trial in detail"],
        ["WiMANS (local partial copy)", "Colab (deleted locally)", "All 3,057 file headers + 77 files in full"],
        ["EHUNAM, WiMANS (full), CSI-Bench", "Colab", "<b>Not audited yet</b> - Phase 0 of the plan"],
    ], [52 * mm, 55 * mm, 67 * mm]))

    # ---------------- 2. Findings per dataset ----------------
    s.append(PageBreak())
    s.append(P("2. Findings, dataset by dataset", H1))

    def ds(name, rows, verdict, colour):
        items = [P(name, H2), table(rows, [38 * mm, 136 * mm], header=False)]
        items.append(Spacer(1, 3))
        items.append(box([P(verdict, BODY)], colour))
        return KeepTogether(items)

    s.append(ds("80 MHz Human Sensing - people-counting folders (Dataset/)", [
        ["What it is", "Meneghello et al., IEEE DataPort (doi 10.21227/xbhv-f125). Only the people-counting folders PC-1a to PC-4a are present."],
        ["Hardware / radio", "Routers with Nexmon CSI firmware. 5 GHz, channel 42, 80 MHz, 802.11ac. One transmit stream, 4 receive antennas. 173 packets/s (published)."],
        ["What the files hold", "One variable per file, csi_buff: complex channel values, 256 frequency bins (242 usable). Rows cycle through the 4 antennas (verified). No timestamps, no signal-strength field."],
        ["Labels", "File name nNN = number of people moving in the room: 0 and 2-10. <b>The 1-person files are missing</b> in all 4 setups, and PC2a_n00 is empty."],
        ["Watch out", "One recording per (setup, count). Half the spectrum needs a documented sign flip. Setup descriptions (set_names.pdf) are not downloaded."],
    ], "<b>Verdict:</b> the best-labelled counting data you have locally. Use it for method development and "
       "held-out-setup testing. It cannot be reproduced on an ESP32-S3 (5 GHz, 80 MHz, 4 antennas).", OK))

    s.append(ds("CountFi - bus passenger counting (CountFi/48)", [
        ["What it is", "Guo et al., WCNC 2025. Your folder is the <b>complete April 2025 release</b>: the page's download button links to <i>48.zip</i>. Licence CC BY-NC 4.0."],
        ["Hardware / radio", "Two Raspberry Pi receivers with Nexmon. 5 GHz, channel 42, 80 MHz, single antenna (all from the files; the page gives no radio details). About 1,000 packets/s."],
        ["What the files hold", "4,740 pcap files of 1,000 packets (~1 s) each. Each packet holds 256 complex values, signal strength, frame type and a microsecond timestamp. 10 files are empty and 4 hold only a header."],
        ["Labels", "Numbered sub-folders 1-8 and 10. The page says every release ships with ground-truth counts and this download contains nothing else, so the folder number is <b>very likely the passenger count</b> (not stated in words)."],
        ["Watch out", "Two kinds of packet are mixed: data frames (45%, clean) and acknowledgement frames (53%, different structure); keep only data frames. Bins 0-1 and 125-131 hold junk values; dropping the standard 14 empty bins fixes it. The <i>head</i> and <i>end</i> receivers record the same seconds, and files are consecutive slices of a trip, so split by trip block, never by file."],
    ], "<b>Verdict:</b> usable counting data (1-8, 10) once the labels are confirmed by the authors. Not "
       "ESP32-reproducible. The paper's 2023 set (<i>613_20People.zip</i>, counts up to 20) is worth "
       "downloading later.", MID))

    s.append(ds("WiCount (WiCount/)", [
        ["What it is", "Zhao et al., CSI-BERT2 (IEEE TMC 2025). Hugging Face RS2002/WiCount."],
        ["Hardware / radio", "<b>ESP32-S3 receiver</b>, 1 antenna, 2.4 GHz, channel 12, 20 MHz, legacy frames. About 83 packets/s measured (100 claimed)."],
        ["What the files hold", "15 CSV files of 60 s. Each row holds 52 complex subcarriers (104 integers; empty slots already removed), signal strength and timestamps. 22 of 28 columns never change."],
        ["Labels", "Folder = people count 0-3. Files per class: 1 / 7 / 6 / 1."],
        ["Watch out", "Recorded in one evening in the order 0 -> 1 -> 2 -> 3 people, so count is tied to elapsed time. The README's I/Q order contradicts Espressif's documentation, and its process1.py has an indexing bug in the imaginary part."],
    ], "<b>Verdict:</b> the only ESP32-S3 counting data. Too small to train on. Use it as the format "
       "reference and as an ESP32-S3 sanity test.", MID))

    s.append(ds("IIT-BHU multi-human activity (IIT-BHU/)", [
        ["What it is", "Gupta et al., IEEE DataPort (doi 10.21227/e6c6-aa21). Activity recognition with 1, 2 or 3 users."],
        ["Hardware / radio", "ESP32 NodeMCU with ESP32-CSI-Tool. 2.4 GHz channel 6, mostly 40 MHz. Irregular 4-52 packets/s (median 28)."],
        ["What the files hold", "925 usable CSV files, 961k packets, 577 minutes. Each row: CSI (384 or 128 values), signal strength, timestamps."],
        ["Labels", "Folder = number of users (1-3). File name = user IDs + activities (bend, clap, move, run, stretch, wave). <b>No empty-room class.</b>"],
        ["Watch out", "11 byte-identical file pairs under different user IDs; 3 empty files; the same users appear in several groups; about 2.5% of rows come from other transmitters or a different Wi-Fi mode."],
    ], "<b>Verdict:</b> ESP32-format data for a 1-3 group-size task or pre-training, after removing duplicates "
       "and filtering rows.", MID))

    s.append(ds("CSI HAR (CSI HAR/)", [
        ["What it is", "Zhuravchak et al., figshare 14386892 (CC BY 4.0). One person, 3 rooms."],
        ["Hardware / radio", "TP-Link TL-WDR4300 routers with Atheros CSI Tool. 5 GHz channel 60, 40 MHz, 2x2 antennas, 114 subcarriers."],
        ["What the files hold", "67,449 packets. Each row: 114 constant index numbers + 4x114 signal strengths + 4x114 phases. The complex values can be recovered exactly. <b>No timestamps</b>, so the packet rate is unknown."],
        ["Labels", "Per packet: walking, sitting, standing, lying, get up, get down, <b>no_person</b>. The label file is one row shorter than the data in every session."],
    ], "<b>Verdict:</b> good for empty-room vs occupied detection and activity recognition. Cannot train a counter (maximum one person).", WARN))

    s.append(ds("Haron98 - WiFi CSI Router Reception (now Colab only)", [
        ["What it is", "Hugging Face Haron98/WIFI-CSI-Dataset. 3 ESP32 receivers listening to one router."],
        ["What the files hold", "1,600 trials x 3 receivers x 100 frames (~4 s, ~25 frames/s). 52 usable subcarriers per frame. Data is clean apart from 598 broken first lines."],
        ["Labels", "<b>label_00-03 are undefined</b> (the README itself says TODO). label_00 looks quieter, but it was recorded last, so this can't define it."],
        ["Watch out", "In the trial checked in detail, receiver dev3 was very weak (values within +/-9). The 3 receivers record the same moment, so keep them together when splitting."],
    ], "<b>Verdict:</b> unlabelled ESP32 data. Becomes useful only if you learn what the labels mean.", WARN))

    s.append(ds("Datasets in Colab - not yet audited", [
        ["EHUNAM", "0-8 people (big gap above 4), several rooms, Raspberry Pi with Nexmon plus Atheros, 2.4 and 5 GHz, .mat files with complex CSI, signal strength and timestamps. Each capture is saved once per receiver (3-4). <b>(Unverified; from your survey PDF.)</b>"],
        ["WiMANS (full)", "0-5 users (0: 594, 1: 3,564, 2-5: 1,782 samples each), 3 rooms, Intel 5300 3x3 antennas, 1 kHz, 3-s samples, labels in annotation.csv. The local copy had no labels, and 16 of its 31 sample groups showed uncorrelated signal over time (unexplained)."],
        ["CSI-Bench", "NeurIPS 2025, 461 hours, 26 rooms, 16 commodity devices <b>including ESP32-S3</b>. Signal strength only (no phase). Tasks: falls, breathing, location, motion source, activity, user ID, proximity. <b>No people-count labels.</b>"],
    ], "<b>Verdict:</b> EHUNAM and WiMANS are your main labelled training sets. CSI-Bench is for pre-training "
       "only. All three need the Phase 0 audit.", SOFT))

    # ---------------- 3. Usability summary ----------------
    s.append(PageBreak())
    s.append(P("3. What is usable for what", H1))
    s.append(table([
        ["Dataset", "Counts available", "ESP32-S3 match", "Role in the plan"],
        ["EHUNAM", "0-8", "No (Nexmon / Atheros)", "Main labelled training"],
        ["WiMANS (full)", "0-5", "No (Intel 5300)", "Main labelled training"],
        ["80 MHz (Dataset/)", "0, 2-10", "No (5 GHz, 4 antennas)", "Labelled training + held-out-setup test"],
        ["CountFi Apr 2025", "1-8, 10 (very likely)", "No (5 GHz, 80 MHz)", "Labelled training (bus setting)"],
        ["WiCount", "0-3", "<b>Yes</b>", "ESP32-S3 format reference + sanity test"],
        ["IIT-BHU", "1-3 (no empty)", "Mostly (ESP32)", "ESP32 pre-training / 1-3 group size"],
        ["CSI HAR", "0-1 (presence)", "No (Atheros)", "Empty vs occupied stage"],
        ["CSI-Bench", "none", "Partly (has ESP32-S3)", "Self-supervised pre-training"],
        ["Haron98", "unknown", "Yes (ESP32)", "Unlabelled pre-training"],
        ["<b>Your own ESP32-S3 data</b>", "0-5+ (you choose)", "<b>Yes</b>", "<b>Fine-tuning and the final test</b>"],
    ], [38 * mm, 34 * mm, 40 * mm, 62 * mm], shade={10: OK}))

    s.append(P("Rules that apply to every dataset", H2))
    s += bullets([
        "<b>Split by recording, setup, room, day or trip - never by random window.</b> Neighbouring windows from "
        "one recording are near-copies, so random splits give falsely high accuracy.",
        "<b>Keep simultaneous receivers on the same side of a split</b>: CountFi head/end, EHUNAM's receivers, Haron98 dev1-3.",
        "<b>Watch for count tied to time.</b> WiCount and Haron98 recorded classes in order, so a model can learn "
        "drift instead of occupancy. Record your own data in shuffled order.",
        "<b>Define windows in seconds, not packets.</b> Packet rates range from 25 per second to 1,000 per second. Mark lost packets as missing; don't invent them.",
        "<b>Drop columns that never change</b> and the empty subcarriers before modelling.",
    ])

    # ---------------- 4. Plan ----------------
    s.append(PageBreak())
    s.append(P("4. The plan", H1))
    s.append(P("Six phases. Phases 0-3 run in Google Colab, because your D: drive has about 18 GB free and 16 GB "
               "of RAM. Phase 4 needs ESP32-S3 boards and a room. Rough durations assume part-time work."))

    phases = [
        ("Phase 0 - Close the gaps (1-2 days, Colab)", [
            "Audit EHUNAM, the full WiMANS and CSI-Bench the same way as the local sets: shapes, subcarrier layout, packet rate, labels per recording, split units.",
            "WiMANS: build the 'users per sample' table from annotation.csv; explain the 16 groups with uncorrelated signal (probably band or room).",
            "EHUNAM: map every recording to its count using Summary.xlsx; group the 3-4 receiver copies of each capture.",
            "Email the CountFi authors to confirm that folder number = passenger count, and ask what '48', 'ms', 'v2' and 'v3' mean.",
            "Optional: download CountFi 613_20People.zip (paper's benchmark set, up to 20 people) and the 80 MHz set_names.pdf.",
        ]),
        ("Phase 1 - One shared input format (2-3 days)", [
            "Choose a format every dataset can be reduced to <b>and</b> an ESP32-S3 can produce: one link, signal strength only, about 52 subcarriers over a 20 MHz slice.",
            "For 80/40 MHz data, cut out one 20 MHz sub-band; for multi-antenna data, use one antenna at a time.",
            "Resample to one rate (e.g. 50 Hz) with 3-5 s windows; mark gaps as missing.",
            "Normalise per recording and per receiver (removes automatic gain changes and device differences).",
            "Also compute simple count cues per window: how much each subcarrier fluctuates, motion energy, and how many independent patterns the signal contains.",
        ]),
        ("Phase 2 - Honest baselines per dataset (3-5 days)", [
            "Train separately on each labelled dataset with held-out setups, rooms or days.",
            "Models: gradient boosting on the simple cues, and a small 1D-CNN / temporal ConvNet on the windows.",
            "Treat counting both as classification and as ordered regression.",
            "Report: average count error, accuracy within +/-1 person, per-count recall.",
            "Outcome: which count ranges are learnable and which features matter.",
        ]),
        ("Phase 3 - Combine datasets (1-2 weeks)", [
            "Pre-train one encoder without labels on everything in the shared format: CSI-Bench (especially its ESP32 parts), Haron98, IIT-BHU and the labelled sets.",
            "Add a count head trained on EHUNAM + WiMANS + 80 MHz + CountFi.",
            "Test by leaving one whole dataset out; try domain adaptation (removing dataset/device identity from the features).",
            "Check on WiCount: zero-shot first, then after a few minutes of fine-tuning.",
        ]),
        ("Phase 4 - Your own ESP32-S3 recordings (1-2 weeks; decides real accuracy)", [
            "Hardware: 1 ESP32-S3 sending fixed traffic at ~100 packets/s (e.g. ESP-NOW or ping) and 2-3 ESP32-S3 receivers, with the same firmware and logging format as WiCount.",
            "Counts 0-5 or more, recorded in <b>shuffled order</b>, several sessions per count, over several days and at least 2 rooms.",
            "Start every session with an empty-room recording.",
            "Include people sitting still and moving.",
            "Ground truth: a timestamped written log or a phone video. Note the room layout and device positions.",
            "Fine-tune the Phase 3 model; test on a day and a room it never saw.",
        ]),
        ("Phase 5 - Run it on the ESP32-S3 (1 week)", [
            "Model under ~100k parameters, quantised to 8-bit (TFLite Micro or ESP-DL), sized for the S3's 512 KB internal + 8 MB external memory.",
            "On-device processing identical to Phase 1: signal strength, normalise, 3-5 s window, plus a rolling empty-room baseline taken at installation.",
            "Two stages: a cheap presence detector first, then the counter. Smooth the output over time so it doesn't flicker.",
            "Measure speed, memory and accuracy on the device, against the Phase 4 test days.",
        ]),
    ]
    for title, items in phases:
        s.append(KeepTogether([P(title, H2)] + checklist(items)))

    # ---------------- 5. Next steps ----------------
    s.append(PageBreak())
    s.append(P("5. What to do this week", H1))
    s += checklist([
        "Open EHUNAM, WiMANS and CSI-Bench in Colab and run the Phase 0 audit notebooks (Claude can generate them).",
        "Send the CountFi authors the label question (contact: jingtao2023 dot guo at connect dot polyu dot hk).",
        "Order or set up 3-4 ESP32-S3 boards and flash the CSI receive / transmit firmware.",
        "Make one 10-minute test recording (empty room, then 1 person) and check it parses like WiCount.",
        "Decide the target count range for deployment (e.g. 0-5). This sets how much of Phase 4 you need.",
    ])

    s.append(P("6. Open questions", H1))
    s.append(table([
        ["Question", "Who / how to resolve"],
        ["Are CountFi folder numbers the passenger counts? What are '48', 'ms', 'v2', 'v3'?", "CountFi authors (email)"],
        ["What do Haron98 label_00-03 mean?", "Dataset uploader on Hugging Face"],
        ["What are the 80 MHz setups PC-1a..4a? Why are the 1-person files absent?", "set_names.pdf on IEEE DataPort"],
        ["Which 52 subcarriers does WiCount keep, and in which I/Q order?", "Compare with your own ESP32-S3 capture"],
        ["What is CSI HAR's packet rate?", "Authors' code (github.com/retsediv/WIFI_CSI_based_HAR)"],
        ["EHUNAM counts, rooms and file layout", "Phase 0 audit in Colab"],
        ["Why do 16 WiMANS sample groups look uncorrelated over time?", "annotation.csv (band / room) in Phase 0"],
    ], [110 * mm, 64 * mm]))

    s.append(P("7. Where everything is", H1))
    s.append(table([
        ["File", "Contents"],
        ["D:/wifi/reports/dataset_audit/DATASET_AUDIT_REPORT.md", "Full audit report"],
        ["FEATURE_DICTIONARY.md", "Every field of every dataset, with meaning and evidence"],
        ["DATA_ACQUISITION_PROTOCOLS.md", "Hardware, radio settings, rooms, labelling per dataset"],
        ["DATASET_COMPARISON.md", "Side-by-side comparison and which datasets can be combined"],
        ["dataset_inventory.csv, label_distribution.csv, compatibility_matrix.csv", "Tables"],
        ["audit_findings.json, audit_errors.log", "Machine-readable findings; problems found"],
        ["scripts/ (01-12)", "Re-runnable audit scripts: py -P -E scripts/&lt;name&gt;.py ..."],
    ], [92 * mm, 82 * mm]))
    s.append(Spacer(1, 6))
    s.append(P("Wiman/ and Haron98_WIFI-CSI-Dataset/ were deleted from D: on 2026-10-19 at your request "
               "(you use them in Colab). D: now has about 17.8 GB free.", SMALL))

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Wi-Fi CSI People Counting - Audit and Plan",
                            author="Dataset audit", subject="CSI datasets for ESP32-S3 people counting")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1])
