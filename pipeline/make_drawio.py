#!/usr/bin/env python3
"""
make_drawio.py  -  writes system_model.drawio and methodology_combined.drawio

Open the files at https://app.diagrams.net (File > Open from > Device) or in the
draw.io desktop app, edit freely, then File > Export as > PNG (width about 3000 px,
transparent background off) and save the export as
    system_model.png            (Fig. 1 of the paper)
    methodology_combined.png    (Fig. 3 of the paper)
so that the LaTeX placeholders pick them up.
"""
import os
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

S = 40  # scale from the plotting coordinates to draw.io pixels

FILL = {"green": "#eaf2e3", "blue": "#e8eef7", "orange": "#fdf1df", "grey": "#f4f4f4"}


class Diagram:
    def __init__(self, name):
        self.name = name
        self.cells = []
        self.n = 1

    def _id(self):
        self.n += 1
        return f"n{self.n}"

    def box(self, x, y, w, h, text, fill="blue", fs=12, bold=False, ident=None):
        i = ident or self._id()
        style = (f"rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor={FILL.get(fill, fill)};"
                 f"strokeColor=#2c3e50;strokeWidth=1.5;fontSize={fs};align=center;verticalAlign=middle;"
                 + ("fontStyle=1;" if bold else ""))
        val = escape(text).replace("\n", "&lt;br&gt;")
        self.cells.append(
            f'<mxCell id="{i}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x*S:.0f}" y="{y*S:.0f}" width="{w*S:.0f}" height="{h*S:.0f}" as="geometry"/></mxCell>')
        return i

    def label(self, x, y, w, h, text, fs=12, color="#2c3e50", bold=False, italic=False, align="center"):
        i = self._id()
        fstyle = (1 if bold else 0) + (2 if italic else 0)
        style = (f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;"
                 f"fontSize={fs};fontColor={color};fontStyle={fstyle};")
        val = escape(text).replace("\n", "&lt;br&gt;")
        self.cells.append(
            f'<mxCell id="{i}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x*S:.0f}" y="{y*S:.0f}" width="{w*S:.0f}" height="{h*S:.0f}" as="geometry"/></mxCell>')
        return i

    def dot(self, x, y, color="#8a8a8a"):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" value="" style="ellipse;fillColor={color};strokeColor={color};" vertex="1" parent="1">'
            f'<mxGeometry x="{x*S:.0f}" y="{y*S:.0f}" width="10" height="10" as="geometry"/></mxCell>')
        return i

    def edge(self, src, tgt, dashed=False, color="#2c3e50", arrow=True, exit_xy=None, entry_xy=None):
        i = self._id()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor={color};strokeWidth=1.6;"
                 + ("dashed=1;" if dashed else "")
                 + ("endArrow=block;endFill=1;" if arrow else "endArrow=none;"))
        if exit_xy:
            style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        if entry_xy:
            style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        self.cells.append(
            f'<mxCell id="{i}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    def line(self, x1, y1, x2, y2, color="#b03a2e", width=1.6):
        i = self._id()
        self.cells.append(
            f'<mxCell id="{i}" style="endArrow=none;html=1;strokeColor={color};strokeWidth={width};" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1*S:.0f}" y="{y1*S:.0f}" as="sourcePoint"/>'
            f'<mxPoint x="{x2*S:.0f}" y="{y2*S:.0f}" as="targetPoint"/></mxGeometry></mxCell>')

    def save(self, path, w=1000, h=520):
        xml = (f'<mxfile host="app.diagrams.net"><diagram name="{self.name}" id="{self.name}">'
               f'<mxGraphModel dx="1400" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
               f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="1169" math="0" shadow="0">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) + '</root></mxGraphModel></diagram></mxfile>')
        ET.fromstring(xml)  # well-formedness check
        open(path, "w", encoding="utf-8").write(xml)
        print("wrote", path, f"({len(self.cells)} cells)")


def system_model(path):
    d = Diagram("System model")
    w, gap = 3.9, 0.6
    xs = [0.2 + i * (w + gap) for i in range(7)]
    y, h = 3.4, 3.1
    texts = [
        ("1  Sender\ncamera or virtual\ncamera (real or\ndeepfake feed)", "green"),
        ("2  Video encoder\nH.264 rate control\n500 to 5000 kbps\n240p to 720p", "orange"),
        ("3  Network channel\nbandwidth drift,\npacket loss, jitter", "orange"),
        ("4  Receiver client\ndecoder and client\nfilters (blur\nsigma 1.5 and 3.0)", "orange"),
        ("5  Face localisation\ndlib HOG, 1.3x crop\nresize 299x299\nnormalise [-1, 1]", "blue"),
        ("6  CallGuard-X\nfine-tuned Xception\n2-way softmax\n(20.8 M params)", "blue"),
        ("7  Decision\nmean p(fake) over\nK <= 10 frames,\nthreshold, warning", "green"),
    ]
    ids = [d.box(x, y, w, h, t, fill=f, fs=13, ident=f"s{k+1}") for k, ((t, f), x) in enumerate(zip(texts, xs))]
    for a, b in zip(ids[:-1], ids[1:]):
        d.edge(a, b)
    # degradation channel annotation
    for k, t in zip((1, 2, 3), ("D_b , D_r", "D_p", "D_sigma")):
        d.label(xs[k], y - 1.0, w, 0.7, t, fs=15, color="#b03a2e", bold=True)
    d.line(xs[1], y - 1.15, xs[3] + w, y - 1.15)
    d.label(xs[1], y - 1.9, 3 * w + 2 * gap, 0.7, "degradation channel modelled in this study (12 conditions)", fs=14, color="#b03a2e")
    # placement lane
    j = d.dot((xs[4] + xs[5] + w) / 2 + 0.0, y + h + 0.9)
    d.edge("s5", j, arrow=False, color="#8a8a8a", exit_xy=(0.5, 1), entry_xy=(0, 0.5))
    d.edge("s6", j, arrow=False, color="#8a8a8a", exit_xy=(0.5, 1), entry_xy=(1, 0.5))
    d.label(xs[4] - 1.5, y + h + 1.3, 2 * w + gap + 3, 0.7, "stages 5 and 6 can be deployed at one of three places (Section XI)", fs=13, bold=True)
    pw = 5.2
    p = [d.box(px, y + h + 2.8, pw, 2.2, t, fill="grey", fs=13)
         for px, t in zip((10.2, 15.9, 21.6),
                          ("Client side\non-device inference,\nno video leaves call",
                           "Edge server\nmedia-server (SFU) tap,\nshared GPU",
                           "Cloud service\nelastic GPUs,\nhighest latency"))]
    for pid in p:
        d.edge(j, pid, arrow=False, color="#8a8a8a", exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.label(0.2, y + h + 3.0, 8.5, 2.6,
            "Evaluation harness:\nstages 1 to 4 are replaced by offline FFmpeg operators applied to FaceForensics++ clips",
            fs=13, italic=True, align="left")
    d.save(path)


def methodology(path):
    d = Diagram("Methodology")

    def lane(y, title):
        d.label(0.2, y, 14, 0.8, title, fs=15, bold=True, align="left")
        d.line(0.2, y + 0.85, 31.8, y + 0.85, color="#c8c8c8", width=1)

    # lane A
    lane(0.2, "A  Data preparation")
    A = [("FaceForensics++ C23\n5 classes x 200\n= 1,000 videos", "green"),
         ("Video-level split\n850 train / 150 val\n(seed 42)", "blue"),
         ("10 frames per video\nevenly spaced", "blue"),
         ("dlib face crop, 1.3x\nresize 299x299\nnormalise [-1, 1]", "blue"),
         ("~8,500 train and\n~1,500 val frames\n0 = real, 1 = fake", "green")]
    wA, gA, yA, hA = 5.6, 0.9, 1.5, 2.2
    ia = [d.box(0.3 + i * (wA + gA), yA, wA, hA, t, fill=f, fs=13, ident=f"a{i+1}") for i, (t, f) in enumerate(A)]
    for a, b in zip(ia[:-1], ia[1:]):
        d.edge(a, b)
    # lane B
    lane(5.0, "B  Detector (Xception) and fine-tuning")
    B = [("Input\n299x299x3", "grey", 2.9),
         ("Entry flow\n2 conv, 3 sep-conv\nblocks (128,256,728)", "blue", 4.5),
         ("Middle flow\n8 x [3 sep-conv, 728]\nresidual", "blue", 4.2),
         ("Exit flow\nsep-conv 1024,\n1536, 2048", "blue", 3.5),
         ("Global avg\npool (2048)", "blue", 2.9),
         ("Dropout p=0.5\nFC 2048 to 2\nsoftmax", "orange", 3.3),
         ("Cross-entropy, Adam\nlr 1e-3, batch 8, 5\nepochs; final weights\nexported (best.pkl)", "green", 4.7)]
    x, yB, hB = 0.3, 6.4, 2.3
    ib = []
    for k, (t, f, wi) in enumerate(B):
        ib.append(d.box(x, yB, wi, hB, t, fill=f, fs=12, ident=f"b{k+1}"))
        x += wi + 0.5
    for a, b in zip(ib[:-1], ib[1:]):
        d.edge(a, b)
    d.edge("a5", "b1", dashed=True, color="#555555", exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.label(11.0, 4.0, 10, 0.7, "frames used for fine-tuning", fs=12, italic=True, color="#555555")
    # lane C
    lane(10.0, "C  Degradation, inference and evaluation")
    C = [("1,000 clean\nvideos", "green", 3.4),
         ("FFmpeg operators\nDb, Dr, Dsigma, Dp\n12 conditions", "orange", 4.6),
         ("12,000\ndegraded videos", "orange", 3.6),
         ("Per-video inference\nmean p(fake)\nK <= 10 frames", "blue", 4.6),
         ("Per-condition metrics\nAUC, EER, Acc@0.5\nn = 1,000 each", "blue", 4.8),
         ("Plots, tables and\ncomparison with\nliterature", "green", 4.2)]
    x, yC, hC = 0.3, 11.3, 2.2
    ic = []
    for k, (t, f, wi) in enumerate(C):
        ic.append(d.box(x, yC, wi, hC, t, fill=f, fs=12, ident=f"c{k+1}"))
        x += wi + 1.0
    for a, b in zip(ic[:-1], ic[1:]):
        d.edge(a, b)
    d.edge("b7", "c4", dashed=True, color="#555555", exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    d.label(24.0, 9.6, 6, 0.7, "checkpoint best.pkl", fs=12, italic=True, color="#555555")
    d.save(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    system_model(os.path.join(out, "system_model.drawio"))
    methodology(os.path.join(out, "methodology_combined.drawio"))
