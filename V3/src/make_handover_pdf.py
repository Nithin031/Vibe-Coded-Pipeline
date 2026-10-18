"""Build the complete V3 collaborator handover PDF.

Run from the repository root:
    py -3.12 V3/src/make_handover_pdf.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reports" / "V3_HANDOVER.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleV3", parent=styles["Title"], fontSize=22,
                          leading=27, alignment=TA_CENTER, spaceAfter=18,
                          textColor=colors.HexColor("#17324d")))
styles.add(ParagraphStyle(name="SubtitleV3", parent=styles["Normal"], fontSize=11,
                          leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#536878"),
                          spaceAfter=24))
styles.add(ParagraphStyle(name="H1V3", parent=styles["Heading1"], fontSize=16,
                          leading=20, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#17324d")))
styles.add(ParagraphStyle(name="H2V3", parent=styles["Heading2"], fontSize=12,
                          leading=15, spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2a5d7c")))
styles.add(ParagraphStyle(name="BodyV3", parent=styles["BodyText"], fontSize=9.3,
                          leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallV3", parent=styles["BodyText"], fontSize=7.5,
                          leading=9.5, spaceAfter=3))
styles.add(ParagraphStyle(name="CodeV3", parent=styles["Code"], fontName="Courier",
                          fontSize=7.5, leading=10, backColor=colors.HexColor("#f1f4f6"),
                          borderPadding=5, spaceBefore=4, spaceAfter=7))


def p(text, style="BodyV3"):
    return Paragraph(text, styles[style])


def bullet(text):
    return p("&bull; " + text)


def table(data, widths=None, small=False):
    converted = []
    for row in data:
        converted.append([cell if hasattr(cell, "wrap") else p(escape(str(cell)), "SmallV3" if small else "BodyV3")
                          for cell in row])
    t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
    ]))
    return t


def image(path, width=6.7 * inch):
    if not path.exists():
        return p(f"Missing figure: {path.name}", "SmallV3")
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        ratio = im.height / max(im.width, 1)
    return Image(str(path), width=width, height=width * ratio)


def load_metrics():
    with (ROOT / "results" / "metrics" / "v3_metrics.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "", "nan")]
    return sum(values) / len(values) if values else None


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#687782"))
    canvas.drawString(0.55 * inch, 0.35 * inch, "V3 ESP32 CSI handover | generated from saved artifacts")
    canvas.drawRightString(7.7 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    metrics = load_metrics()
    models = json.loads((ROOT / "manifests" / "models.json").read_text())
    story = []

    story += [p("V3 ESP32 WiFi CSI Occupancy Detection", "TitleV3"),
              p("Complete project handover for a new collaborator", "SubtitleV3"),
              p("Purpose", "H1V3"),
              p("V3 is a research pipeline for detecting room presence and motion state from ESP32 WiFi Channel State Information under room, hardware, and measured WiFi-channel changes. This PDF summarizes the implementation, approved model files, saved evidence, reproduction commands, and remaining work.")]
    story += [p("Executive summary", "H1V3"),
              bullet("Presence task: empty versus occupied."),
              bullet("Motion task: empty, stationary person, or moving person."),
              bullet("Common deep input: float16 relative-amplitude sequence [32, 60] plus a bool[60] packet-gap mask."),
              bullet("Sampling is irregular at acquisition; 20 Hz is a preprocessing grid, not a native ESP32 rate."),
              bullet("The strongest V3 evidence is real room transfer, hardware variation, and a measured congested-channel test."),
              bullet("A 30-second empty-room calibration improves the N3 held-out-room presence result from about 0.47 to 0.72 balanced accuracy and reduces false alarms from 100% to about 10% for the reported ensemble."),
              bullet("The final deployment files were trained after the historical held-out runs and are not the exact evaluation checkpoints."),
              PageBreak()]

    story += [p("1. End-to-end system", "H1V3"),
              p("A transmitter ESP32 sends regular WiFi packets. A receiver ESP32 measures CSI for packets it receives and streams records over USB serial. The host parses the records, converts I/Q to amplitude, filters unstable bins, resamples the irregular stream to 20 Hz, creates 3-second windows every second, and runs LightGBM, CNN, or transformer inference. Calibration and five-window smoothing are separate runtime/evaluation operations."),
              p("Data flow", "H2V3"),
              table([["Stage", "Implementation", "Output"],
                     ["Acquisition", "Two ESP32 boards; receiver USB serial", "CSI_DATA records"],
                     ["Parsing", "esp32_io.py: parse_raw_lines, lltf_amplitude", "timestamp, RSSI, L-LTF amplitude"],
                     ["Resampling", "preprocess.py: to_uniform_grid", "20 Hz signal plus missing mask"],
                     ["Windowing", "build.py: stream_windows", "3 s / 1 s-hop windows"],
                     ["Representation", "features.py: relative_fluctuation, cnn_input", "32 bins x 60 samples"],
                     ["Prediction", "deep.py or LightGBM", "presence or motion logits"],
                     ["Postprocessing", "calibrate_v3.py and experiments_noise.py", "calibrated/smoothed label"]],
                    [1.25 * inch, 2.9 * inch, 2.55 * inch], small=True),
              p("The receiver parser supports 104, 128, or 384 interleaved I/Q values and retains the L-LTF portion. The firmware documentation also contains an amplitude-only capture format; that format is not sufficient for the complete V3 timestamp/RSSI/IQ contract without an adapter."),
              p("Hardware requirements", "H2V3"),
              bullet("Two ESP32 DevKit boards, two USB cables, and a laptop or Raspberry Pi host."),
              bullet("ESP-IDF for flashing; receiver/transmitter firmware under datasets/esp32_csi_pipeline/firmware/."),
              bullet("Receiver serial at 460800 baud; capture dependency pyserial>=3.5."),
              bullet("Keep board positions, channel, bandwidth, and link arrangement fixed during calibration and inference."),
              PageBreak()]

    story += [p("2. Preprocessing and features", "H1V3"),
              p("For each packet, the parser retains timestamp, RSSI, CSI length, metadata, and raw interleaved I/Q. Amplitude is computed as |H| = sqrt(I^2 + Q^2). valid_subcarriers removes bins whose median amplitude is below 0.15 or above 4.0 times the typical level. relative_fluctuation divides each subcarrier by its window median, subtracts one, and clips robust outliers at six MADs."),
              p("The irregular packet stream is mapped to floor(time*20) bins. Packets sharing a bin are averaged. Empty bins are linearly interpolated, but missing=True is retained. build.py then creates 60-sample windows every 20 samples and resamples surviving subcarriers to 32 bins. This preserves a diagnostic gap mask but loses native packet timing detail beyond the 20 Hz grid."),
              p("Feature families implemented for LightGBM", "H2V3"),
              table([["Group", "Implemented features", "Current status"],
                     ["Motion", "std, robust std/MAD, first-difference energy, range, quantiles", "Current engineered baseline"],
                     ["PCA/shape", "top eigen-energy, effective rank, participation, correlations", "Implemented; not separately ablated"],
                     ["Spectrum", "band powers, centroid, entropy, autocorrelation", "Implemented; needs grouped ablation"],
                     ["Periodicity", "peak share, flatness, periodic ACF, peak frequency", "Implemented; hypothesis for fans/AC"],
                     ["RSSI", "standard deviation, range, absolute differences", "Implemented dynamics only; no absolute RSSI"],
                     ["Sequence", "relative amplitude [32,60]", "CNN and transformer input"],
                     ["Phase/IQ", "raw archive retained where available", "Not used by V3 models"]],
                    [1.0 * inch, 3.65 * inch, 2.05 * inch], small=True),
              p("The transformer consumes the [60] gap mask as learned padding tokens. CNN1D.forward accepts gap but ignores it. LightGBM receives engineered features only."),
              PageBreak()]

    story += [p("3. Approved V3 model files", "H1V3"),
              p("Use only these six classifier artifacts for V3 inference. They are state dictionaries/LightGBM boosters and require the matching definitions in src/deep.py or LightGBM."),
              table([["Task", "Family", "File", "Classes", "Load"],
                     ["Presence", "LightGBM", "models/presence_lgbm.txt", "empty, occupied", "lgb.Booster(model_file=...)"],
                     ["Presence", "CNN", "models/presence_cnn.pt", "empty, occupied", "deep.make_model(\"cnn\", n_classes=2)"],
                     ["Presence", "Transformer", "models/presence_tf.pt", "empty, occupied", "deep.make_model(\"tf\", n_classes=2)"],
                     ["Motion", "LightGBM", "models/motion_lgbm.txt", "empty, static, moving", "lgb.Booster(model_file=...)"],
                     ["Motion", "CNN", "models/motion_cnn.pt", "empty, static, moving", "deep.make_model(\"cnn\", n_classes=3)"],
                     ["Motion", "Transformer", "models/motion_tf.pt", "empty, static, moving", "deep.make_model(\"tf\", n_classes=3)"]],
                    [0.8 * inch, 0.8 * inch, 2.05 * inch, 1.4 * inch, 2.0 * inch], small=True),
              p("V3 also contains models/transformer_pretrained.pt. It is initialization-only, has no occupancy/motion head, and must not be used as a standalone runtime model. V2/models and V4/models are separate-version artifacts."),
              p("The registry load-tested 7/7 V3 .pt/support artifacts. Exact sizes and SHA-256 values are in manifests/models.json."),
              p("Model-to-result mapping", "H2V3"),
              table([["Runtime file", "Saved JSON variant", "Task/result scope"],
                     ["presence_lgbm.txt", "lgbm", "Presence N1-N4 and calibrated counterparts where available"],
                     ["presence_cnn.pt", "cnn", "Presence N1-N4 and calibrated counterparts where available"],
                     ["presence_tf.pt", "tf", "Presence N1-N4 and calibrated counterparts where available"],
                     ["motion_lgbm.txt", "lgbm", "Motion N1-N3 and calibrated counterparts where available"],
                     ["motion_cnn.pt", "cnn", "Motion N1-N3 and calibrated counterparts where available"],
                     ["motion_tf.pt", "tf", "Motion N1-N3 and calibrated counterparts where available"]],
                    [1.8 * inch, 1.0 * inch, 4.25 * inch], small=True),
              p("Ensemble rows are probability averages from evaluation and do not correspond to a separate checkpoint. Calibration changes inputs at runtime/evaluation; it does not create separate weights."),
              PageBreak()]

    story += [p("4. Datasets and protocols", "H1V3"),
              table([["Dataset", "Contribution", "Current use"],
                     ["ESPectre", "Real homes, rooms, multiple ESP32 chip types", "N1/N1c and final V3 training"],
                     ["esp32-csi-pipeline", "Two rooms, quiet and congested channel recordings", "N2/N2c/N3/N3c"],
                     ["WiCount", "ESP32-S3 labelled people-count CSV", "Reference/N4 source"],
                     ["WiGesture/WiFall", "ESP32 activity recordings", "Transformer masked pretraining"],
                     ["OTH thesis", "ESP-NOW HT40 labelled reference", "N4 source/reference"],
                     ["OpenCSI", "Bursty mesh-link recordings", "Excluded from V3 scores"],
                     ["HomeOccupancy", "ESP32-C6 home empty/sleep/work", "Separate V4 fine-tuning"]],
                    [1.25 * inch, 3.2 * inch, 2.6 * inch], small=True),
              p("Protocols", "H2V3"),
              bullet("N1: leave one ESPectre room out; train on the other rooms."),
              bullet("N2: train on quiet pipeline recording; test same room on measured congested channel."),
              bullet("N3: train pipeline room 1; test unseen room 2."),
              bullet("N4: train on other ESP32 data; test pipeline setup."),
              bullet("N1c/N2c/N3c: corresponding protocols with a 30-second empty-room reference; reference windows are excluded from scoring."),
              p("The calibration site key is dataset + room + chip + channel. Energy features are log-ratio calibrated and each sequence bin is divided by an empty-reference robust level. Five-window median smoothing is applied within recordings for smoothed scores."),
              p("Historical source files", "H2V3"),
              p("N1_espectre_leave_room_out.json, N1c_espectre_leave_room_out_calibrated.json, N2_pipeline_quiet_to_congested.json, N2c_pipeline_quiet_to_congested_calibrated.json, N3_pipeline_room1_to_room2.json, N3c_pipeline_room1_to_room2_calibrated.json, and N4_all_to_pipeline.json are under results/. The canonical flattened table is results/metrics/v3_metrics.csv."),
              PageBreak()]

    story += [p("5. Verified results", "H1V3"),
              p("The table below is computed from the saved canonical CSV. It summarizes smoothed presence rows for the principal protocols; the full CSV retains every model, task, room, window/smoothed value, and confusion matrix."),
              table([["Protocol", "Model", "Balanced accuracy", "False alarm", "Occupied sensitivity"],
                     ["N2 raw", "Transformer", "0.937", "0.016", "0.984"],
                     ["N2 calibrated", "Ensemble", "0.910", "0.028", "0.992"],
                     ["N3 raw", "Ensemble", "0.472", "1.000", "1.000"],
                     ["N3 calibrated", "Ensemble", "0.719", "0.095", "0.991"],
                     ["N4 raw", "Transformer", "0.674", "0.647", "0.994"]],
                    [1.15 * inch, 1.15 * inch, 1.55 * inch, 1.25 * inch, 2.0 * inch], small=True),
              p("These headline values are source-result values, not estimates from prose. N3 demonstrates the central V3 problem: an unseen room causes near-universal empty-room false alarms without calibration, while a short empty-room reference substantially improves the ensemble. Calibration is helpful but not universally sufficient; N1 varies by held-out home room."),
              p("Figures", "H2V3"),
              image(ROOT / "results" / "figures" / "fig1_presence_raw_vs_cal.png"),
              p("Figure 1: presence balanced accuracy, raw versus calibrated.", "SmallV3"),
              image(ROOT / "results" / "figures" / "fig2_false_alarms.png"),
              p("Figure 2: empty-room false-alarm rate.", "SmallV3"),
              PageBreak()]

    for name, caption in [
        ("fig3_congested_channel.png", "N2 confusion matrices on a measured congested channel."),
        ("fig4_espectre_rooms.png", "N1 held-out ESPectre-room comparison."),
        ("fig5_empty_vs_occupied.png", "Empty versus occupied fluctuation distributions."),
        ("fig6_n3_cross_room.png", "N3 unseen-room raw versus calibrated comparison."),
        ("fig7_n4_cross_dataset.png", "N4 new-setup transfer comparison."),
        ("fig8_model_family_n2.png", "N2 model-family comparison."),
        ("fig9_presence_per_class.png", "Presence empty specificity and occupied sensitivity."),
    ]:
        story += [image(ROOT / "results" / "figures" / name), p(caption, "SmallV3")]
    story += [PageBreak()]

    story += [p("6. Reproduction and use", "H1V3"),
              p("From the repository root, with Python 3.12 and requirements installed:") ,
              p("py -3.12 V3/src/show_transformer.py --task presence\npy -3.12 V3/src/show_transformer.py --task motion", "CodeV3"),
              p("These commands use cached pipeline windows and perform offline checkpoint loading plus one-window inference. They do not access an ESP32 and are not a live serial application."),
              p("Regenerate figures and registries:") ,
              p("py -3.12 V3/src/plots.py\npy -3.12 V3/src/dashboard_plots.py\npy -3.12 V3/src/build_registry.py", "CodeV3"),
              p("Run historical evaluations from V3/src only when intentionally retraining/evaluating:") ,
              p("py -3.12 experiments_noise.py N1 N2 N3 N4\npy -3.12 experiments_noise.py N1c N2c N3c", "CodeV3"),
              p("Final full-pool model training, which can overwrite final filenames, is:") ,
              p("py -3.12 V3/src/train_final_v3.py --epochs 30", "CodeV3"),
              p("Model loading example:") ,
              p("import sys, torch\nsys.path.insert(0, 'V3/src')\nimport deep\nstate = torch.load('V3/models/presence_tf.pt', map_location='cpu')\nmodel = deep.make_model('tf', n_classes=2)\nmodel.load_state_dict(state)", "CodeV3"),
              p("For live capture, use the firmware and capture documentation under datasets/esp32_csi_pipeline/. The raw receiver schema must be checked first; complete live serial-to-model inference and Raspberry Pi latency have not been validated."),
              p("Software", "H2V3"),
              bullet("Python 3.12 was used for current Windows validation."),
              bullet("Root requirements include numpy, pandas, scipy, scikit-learn, lightgbm, matplotlib, reportlab, and torch."),
              bullet("Capture requires pyserial>=3.5 and ESP-IDF only for flashing."),
              PageBreak()]

    story += [p("7. Limitations and next actions", "H1V3"),
              bullet("The CNN ignores the gap mask even though preprocessing produces it."),
              bullet("Missing bins are interpolated, so the gap mask is required for interpretation."),
              bullet("Calibrated paths need explicit DataFrame/NumPy index-alignment tests after filtering."),
              bullet("Overlapping windows make random window splits invalid for generalization."),
              bullet("Historical results do not include event-level smoothing delay or confidence intervals."),
              bullet("Phase, complex CSI, and proposed feature groups have not been empirically validated."),
              bullet("Live receiver format compatibility and Raspberry Pi real-time latency remain unvalidated."),
              p("Recommended order", "H2V3"),
              table([["Priority", "Action", "Reason"],
                     ["1", "Test raw CSI_DATA schema, timestamp wrap, malformed packets, and gap masks", "Prevents invalid live input"],
                     ["2", "Assert index alignment in calibrated experiment paths", "Protects new ablations from silent misalignment"],
                     ["3", "Run grouped feature ablations A-G", "Identify features that generalize beyond training rooms"],
                     ["4", "Run controlled V4 HomeOccupancy fine-tuning", "Test temporal home-domain adaptation separately"],
                     ["5", "Validate live hardware and measure latency", "Required before deployment claims"]],
                    [0.65 * inch, 3.3 * inch, 3.3 * inch], small=True),
              p("Source documents", "H2V3"),
              p("V3_HANDOVER.md, QUICKSTART.md, MODEL_AND_RESULTS_INDEX.md, NEXT_STEPS.md, V3_REPORT.md, PROTOCOL.md, FEATURE_INVENTORY.md, LIMITATIONS.md, and protocol_feature_audit.md are under V3/docs/. The registry and cleanup records are under V3/manifests/ and V3/results/reports/."),
              p("End of handover", "SubtitleV3")]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=0.55 * inch,
                            leftMargin=0.55 * inch, topMargin=0.55 * inch,
                            bottomMargin=0.55 * inch, title="V3 ESP32 CSI Handover",
                            author="V3 project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()