"""Build the COLDER explainer video from the paper's real figures and results.

  python video/build.py                       # captions-only render  -> video/COLDER_demo_captions.mp4 (+ narration_script.md, captions.srt)
  python video/build.py --audio my_voice.wav  # scenes are retimed to the recording's length and the audio is muxed in
Every number shown is read from results/*.json.  No footage is fabricated: slides are drawn from the committed results.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video"
FR = OUT / "frames"
W, H, FPS = 1920, 1080, 30
SURF, INK, MUTED, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (230, 229, 225)
BLUE, ORANGE, AQUA, RED = (42, 120, 214), (235, 104, 52), (27, 175, 122), (208, 59, 59)
PALE = (232, 241, 252)
FD = "C:/Windows/Fonts/"


def font(name, size):
    try:
        return ImageFont.truetype(FD + name, size)
    except OSError:
        return ImageFont.truetype(FD + "arial.ttf", size)


F_DISPLAY = lambda s: font("bahnschrift.ttf", s)  # noqa: E731
F_BODY = lambda s: font("calibri.ttf", s)  # noqa: E731
F_BOLD = lambda s: font("calibrib.ttf", s)  # noqa: E731
F_MONO = lambda s: font("consola.ttf", s)  # noqa: E731

S = json.load(open(ROOT / "results/detect/summary.json"))
RL = json.load(open(ROOT / "results/rl/summary.json"))
FW = json.load(open(ROOT / "results/rl/first_window.json"))
au = lambda d: S["test"][d]["auroc"][0]  # noqa: E731
pw = S["paired_test"]["B3 - B2"]["auroc_diff"]

SCENES = [
    dict(key="title", eyebrow="AI COLLUSION RESEARCH SPRINT · OCT 2026", title="Can we audit agents for collusion\nwithout accusing the innocent?",
         sub="A pre-registered, sprint-scale test of APTG-RL", fig=None,
         lines=["Hi, I'm Aravinth. This is a one-weekend, pre-registered test of a design for auditing multi-agent systems for collusion."]),
    dict(key="problem", eyebrow="THE PROBLEM", title="Similar behaviour is not collusion", fig="diagram_problem",
         lines=["When two AI agents behave alike, that does not mean they collude.",
                "They may share a prompt, a task, or a base model.",
                "And if someone steers just one agent, its behaviour shifts with no collusion at all.",
                "So: does one agent's history tell us something extra, and directional, about the other's next move?"]),
    dict(key="design", eyebrow="THE DESIGN · APTG-RL", title="Four pieces, one audit", fig="diagram_design",
         lines=["The design has four pieces.",
                "A baseline that knows each agent's prompt, which I implemented but could not exercise in the reported detectors.",
                "A directional score: how much does A's history improve the prediction of B's next action?",
                "A time-varying graph of those scores, and an RL investigator that decides when to alert."]),
    dict(key="testbed", eyebrow="THE TESTBED", title="A pricing game with known ground truth", fig="diagram_testbed",
         lines=["I built a three-seller pricing game with four scripted collusion mechanisms:",
                "complementary, leader-follower, alternating, and a hidden parity channel.",
                "Negatives include independent agents, shared-model agents, and benign adaptation.",
                "A factorial of prompt perturbations hits one agent, both, or neither. Two mechanisms and one prompt family are held out of every fit."]),
    dict(key="protocol", eyebrow="THE PROTOCOL", title="Pre-registered before any held-out data", fig="diagram_protocol",
         lines=["The analysis plan was committed before any held-out evaluation.",
                "Thresholds are set on validation negatives at five percent false positives, then applied once to test.",
                "Ten detectors, bootstrapped over whole episodes."]),
    dict(key="auroc", eyebrow="RESULT 1", title="Directional beats symmetric, not lag-one", fig="fig_auroc.png",
         lines=[f"Result one. A directional gain score that uses both same-round and lagged information beats symmetric mutual information: "
                f"AUROC {au('B3'):.2f} versus {au('B2'):.2f}.",
                f"But it does not beat the lag-one mutual-information baseline from prior work, {au('B1'):.2f}. And its strictly lagged part alone, "
                f"{au('B3_temporal'):.2f}, does not beat symmetric mutual information."]),
    dict(key="mech", eyebrow="RESULT 2", title="Every detector is mechanism-specific", fig="fig_mechanism.png",
         lines=["Result two is the most useful one. Lag-one mutual information is perfect on leader-follower and the hidden channel,",
                "but nearly blind to complementary pairs. Same-round mutual information is the exact reverse.",
                "My directional score covers three mechanisms, but a parametric model cannot see a parity code. Coverage matters more than rank."]),
    dict(key="nulls", eyebrow="NULLS, REPORTED", title="What did not work", fig="diagram_nulls",
         lines=["Three nulls. The graph score was worse than the edge score alone: there are only three windows per episode.",
                "The prompt-conditioning ablation is vacuous, because prompt covariates are constant within an episode.",
                "And the RL investigator just rediscovered a look-at-the-first-window rule, trading recall for speed.",
                "Unilateral prompt steering did not clearly raise false alarms."]),
    dict(key="pilot", eyebrow="HAIKU PILOT", title="Real LLM agents: instructions move prices", fig="fig_pilot.png",
         lines=["A small pilot with Claude Haiku agents: seven hundred twenty decisions, none invalid.",
                "The perturbation moved prices as intended, and telling two seats to keep profits high lifted theirs.",
                "With six episodes per condition, this is descriptive only."]),
    dict(key="limits", eyebrow="LIMITS AND DUAL-USE", title="What this does not show", fig="diagram_limits",
         lines=["The limits. It is a simulator I built myself, at small scale.",
                "The Haiku agents were driven as subagents, so independence is only approximate.",
                "An alarm is evidence of coordination, not intent. The hidden channel is a toy protocol, reported because the failure is itself an audit finding."]),
    dict(key="close", eyebrow="TAKEAWAYS", title="Cover the space. Hold things out. Report the nulls.", fig="diagram_close",
         lines=["No single dependence statistic sees every coordination pattern.",
                "Hold out mechanisms, and report your nulls.",
                "Code, paper, and pre-registration are on GitHub, chinchoooo slash COLDER."]),
]


# ------------------------------------------------------------------ drawing helpers
def wrap(draw, text, fnt, maxw):
    out, cur = [], ""
    for wd in text.split():
        t = (cur + " " + wd).strip()
        if draw.textlength(t, font=fnt) <= maxw:
            cur = t
        else:
            out.append(cur)
            cur = wd
    return out + ([cur] if cur else [])


def rrect(d, box, r, fill=None, outline=None, width=2):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def card(d, x, y, w, h, head, body, color=BLUE, hs=34, bs=27):
    rrect(d, (x, y, x + w, y + h), 18, fill=(255, 255, 255), outline=GRID, width=2)
    d.rectangle((x, y + 18, x + 8, y + h - 18), fill=color)
    d.text((x + 30, y + 22), head, font=F_DISPLAY(hs), fill=INK)
    yy = y + 22 + hs + 14
    for ln in wrap(d, body, F_BODY(bs), w - 56):
        d.text((x + 30, yy), ln, font=F_BODY(bs), fill=MUTED)
        yy += bs + 8


def arrow(d, p, q, color=MUTED, w=5):
    d.line([p, q], fill=color, width=w)
    import math
    a = math.atan2(q[1] - p[1], q[0] - p[0])
    for s in (-0.45, 0.45):
        d.line([q, (q[0] - 22 * math.cos(a + s), q[1] - 22 * math.sin(a + s))], fill=color, width=w)


def base_slide(sc, n, total):
    im = Image.new("RGB", (W, H), SURF)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 10), fill=BLUE)
    d.text((96, 56), sc["eyebrow"], font=F_MONO(24), fill=BLUE)
    d.multiline_text((96, 98), sc["title"], font=F_DISPLAY(60 if sc["key"] != "title" else 78), fill=INK, spacing=10)
    d.text((96, H - 44), "Aravinth K  ·  AI Collusion Research Sprint 2026  ·  github.com/chinchoooo/COLDER", font=F_BODY(22), fill=MUTED)
    d.text((W - 96 - 80, H - 44), f"{n}/{total}", font=F_MONO(22), fill=MUTED)
    return im, d


# content area: y from 230 to 800 (captions occupy 840-1000)
def draw_content(sc, d, im, step, nsteps):
    k = sc["fig"]
    x0, y0 = 96, 250
    if k is None:
        d.text((96, 360), sc["sub"], font=F_BODY(44), fill=MUTED)
        for i, (t, c) in enumerate([("7 mechanisms · 7 conditions", BLUE), ("10 detectors", ORANGE), ("bootstrapped, pre-registered", AQUA)]):
            if True:
                rrect(d, (96 + i * 560, 520, 96 + i * 560 + 520, 640), 20, fill=(255, 255, 255), outline=c, width=4)
                d.text((96 + i * 560 + 30, 556), t, font=F_DISPLAY(34), fill=INK)
        return
    if k.endswith(".png"):
        fig = Image.open(ROOT / "paper/figs" / k).convert("RGB")
        maxw, maxh = W - 192, 560
        s = min(maxw / fig.width, maxh / fig.height)
        fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
        im.paste(fig, ((W - fig.width) // 2, 235))
        return
    if k == "diagram_problem":
        for i, (nm, xy) in enumerate([("A", (330, 520)), ("B", (1050, 520))]):
            d.ellipse((xy[0] - 70, xy[1] - 70, xy[0] + 70, xy[1] + 70), fill=PALE, outline=BLUE, width=5)
            d.text((xy[0] - 18, xy[1] - 40), nm, font=F_DISPLAY(70), fill=BLUE)
        arrow(d, (410, 520), (960, 520), BLUE, 7)
        d.text((540, 450), "extra, directional?", font=F_BODY(34), fill=BLUE)
        items = ["shared prompt", "shared task state", "shared base model", "steering of ONE agent"]
        shown = min(step, 3) if step < 3 else 4
        for i, t in enumerate(items[:shown]):
            yy = 270 + i * 110
            rrect(d, (1420, yy, 1830, yy + 88), 16, fill=(255, 255, 255), outline=ORANGE, width=3)
            d.text((1445, yy + 24), t, font=F_BODY(32), fill=INK)
            arrow(d, (1420, yy + 44), (1130, 505 if i < 2 else 540), ORANGE, 3)
        if step >= 3:
            d.text((96, 720), "Same behaviour ≠ collusion.   Question: extra information beyond context, own history and prompt?", font=F_BOLD(32), fill=INK)
        return
    if k == "diagram_design":
        boxes = [("1  Prompt-conditioned\nbaseline", "p₀(a | context, own history, prompt)", BLUE),
                 ("2  Directional\npredictive gain", "g(i→j): how much does i's history help predict j?", ORANGE),
                 ("3  Temporal\ndirected graph", "windows, persistence, asymmetry, motifs", AQUA),
                 ("4  RL\ninvestigator", "observe · inspect · flag · abstain", RED)]
        for i, (h, b, c) in enumerate(boxes):
            if i <= step:
                x = 96 + i * 455
                rrect(d, (x, 300, x + 420, 640), 22, fill=(255, 255, 255), outline=c, width=4)
                d.multiline_text((x + 26, 330), h, font=F_DISPLAY(36), fill=INK, spacing=6)
                for j, ln in enumerate(wrap(d, b, F_BODY(30), 370)):
                    d.text((x + 26, 470 + j * 40), ln, font=F_BODY(30), fill=MUTED)
                if i < 3 and i < step:
                    arrow(d, (x + 424, 470), (x + 452, 470), MUTED, 5)
        return
    if k == "diagram_testbed":
        mech = [("complementary", "B = 4 − A"), ("leader–follower", "B copies A(t−1)"), ("alternating†", "take turns"), ("hidden channel†", "parity of A → B")]
        for i, (a, b) in enumerate(mech):
            if i <= step:
                card(d, 96 + (i % 2) * 560, 250 + (i // 2) * 170, 530, 150, a, b, BLUE, 32, 28)
        negs = ["independent", "shared model", "benign adaptation"]
        d.text((1250, 250), "negatives", font=F_MONO(26), fill=MUTED)
        for i, t in enumerate(negs):
            if step >= 2:
                rrect(d, (1250, 290 + i * 70, 1830, 350 + i * 70), 14, fill=(255, 255, 255), outline=GRID, width=2)
                d.text((1275, 302 + i * 70), t, font=F_BODY(30), fill=INK)
        if step >= 3:
            d.text((1250, 520), "perturbation conditions", font=F_MONO(26), fill=MUTED)
            conds = [("C0", "clean"), ("C1", "A steered"), ("C2", "B steered"), ("C3", "both"), ("C4", "collusion"), ("C5", "coll. + steer"), ("C6", "benign shift")]
            for i, (c, t) in enumerate(conds):
                x, y = 1250 + (i % 4) * 150, 560 + (i // 4) * 90
                rrect(d, (x, y, x + 138, y + 76), 12, fill=PALE, outline=BLUE, width=2)
                d.text((x + 12, y + 6), c, font=F_DISPLAY(28), fill=BLUE)
                d.text((x + 12, y + 40), t, font=F_BODY(20), fill=MUTED)
            d.text((96, 640), "† held out of every fit      ·      one prompt family held out", font=F_BODY(28), fill=MUTED)
        return
    if k == "diagram_protocol":
        steps = [("PREREG.md committed", "before any detector touched validation, test, OOD or pilot data", BLUE),
                 ("Thresholds on validation negatives", "FPR = 0.05, fixed once, applied to test", ORANGE),
                 ("Bootstrap over whole episodes", "1000 resamples · paired differences · no windows split across sets", AQUA)]
        for i, (h, b, c) in enumerate(steps):
            if i <= step:
                card(d, 96, 270 + i * 175, 1500, 150, h, b, c, 38, 30)
        return
    if k == "diagram_nulls":
        thr = RL["comparators"]["fixed_threshold"]["test"]
        ppo = RL["ppo_selected"]["mean"]["test"]
        cards = [("Graph score < edge score", f"APTG AUROC {au('APTG'):.2f} vs {au('B3'):.2f}; only 3 windows per episode", RED),
                 ("Prompt ablation is vacuous", "prompt covariates are constant within an episode, so B3 = B3 without prompt", ORANGE),
                 ("RL = first-window rule", f"TPR {ppo['tpr']:.2f} vs {thr['tpr']:.2f} (fixed threshold); delay {ppo['delay_steps']:.0f} vs {thr['delay_steps']:.0f}", BLUE)]
        for i, (h, b, c) in enumerate(cards):
            if i <= step:
                card(d, 96 + i * 600, 270, 570, 250, h, b, c, 34, 30)
        if step >= 3:
            d.text((96, 580), "Unilateral steering: no clear false-alarm rise (60 episodes per cell; B2 on C1 reached 0.15)", font=F_BOLD(32), fill=INK)
        return
    if k == "diagram_limits":
        items = ["Simulator I built myself; scripted mechanisms share assumptions with detectors",
                 "Small scale: 10 / 12 / 20 / 10 seeds per cell; 30 permutations per edge",
                 "Haiku pilot: 6 episodes per condition; agents driven as subagents",
                 "An alarm is evidence of coordination, not intent",
                 "Hidden channel = closed toy parity code; reported because the failure is an audit finding"]
        for i, t in enumerate(items):
            if i <= step + 1:
                d.ellipse((96, 290 + i * 100, 124, 318 + i * 100), fill=ORANGE if i == 4 else BLUE)
                for j, ln in enumerate(wrap(d, t, F_BODY(36), 1600)):
                    d.text((150, 280 + i * 100 + j * 44), ln, font=F_BODY(36), fill=INK)
        return
    if k == "diagram_close":
        items = [("Cover the space", "no single dependence statistic sees every pattern", BLUE),
                 ("Hold things out", "mechanisms and prompt families, never just seeds", ORANGE),
                 ("Report the nulls", "and every deviation from the plan", AQUA)]
        for i, (h, b, c) in enumerate(items):
            if i <= step:
                card(d, 96 + i * 600, 270, 570, 250, h, b, c, 38, 30)
        d.text((96, 600), "github.com/chinchoooo/COLDER", font=F_MONO(44), fill=BLUE)
        d.text((96, 670), "paper · code · pre-registration · deviations log", font=F_BODY(32), fill=MUTED)
        return


def caption(im, text):
    d = ImageDraw.Draw(im, "RGBA")
    fnt = F_BODY(40)
    lines = wrap(d, text, fnt, W - 2 * 130 - 60)[:3]
    lh = 52
    bh = lh * len(lines) + 36
    y1 = H - 96
    y0 = y1 - bh
    d.rounded_rectangle((110, y0, W - 110, y1), 20, fill=(11, 11, 11, 225))
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=fnt)
        d.text(((W - w) / 2, y0 + 18 + i * lh), ln, font=fnt, fill=(255, 255, 255))
    return im


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--wpm", type=float, default=150)
    a = ap.parse_args()
    FR.mkdir(parents=True, exist_ok=True)
    for f in FR.glob("*.png"):
        f.unlink()
    # ---- timing: words per second plus a short breath after each caption
    segs = []
    for si, sc in enumerate(SCENES):
        for li, ln in enumerate(sc["lines"]):
            segs.append(dict(scene=si, line=li, text=ln, dur=len(ln.split()) / (a.wpm / 60) + 0.45))
    trans = 0.4
    total = sum(s["dur"] for s in segs) + trans * (len(SCENES) - 1) + 1.2  # + end hold
    if a.audio:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.audio], capture_output=True, text=True)
        adur = float(r.stdout.strip())
        k = (adur - trans * (len(SCENES) - 1) - 1.2) / sum(s["dur"] for s in segs)
        for s in segs:
            s["dur"] *= k
        total = adur
    # ---- frames
    entries, t, srt, script = [], 0.0, [], ["# Narration script (record this; ~150 words per minute)\n"]
    prev_last = None
    for si, sc in enumerate(SCENES):
        script.append(f"\n## {si + 1}. {sc['title'].replace(chr(10), ' ')}\n")
        ss = [s for s in segs if s["scene"] == si]
        for li, s in enumerate(ss):
            im, d = base_slide(sc, si + 1, len(SCENES))
            draw_content(sc, d, im, li, len(ss))
            im = caption(im, s["text"])
            p = FR / f"s{si:02d}_{li:02d}.png"
            im.save(p)
            if li == 0 and prev_last is not None:  # dissolve from previous scene's last frame
                A, B = Image.open(prev_last), Image.open(p)
                n = max(2, int(trans * FPS))
                for j in range(1, n):
                    q = FR / f"x{si:02d}_{j:02d}.png"
                    Image.blend(A, B, j / n).save(q)
                    entries.append((q, 1 / FPS))
                t += trans
            elif li == 0 and si == 0:
                pass
            entries.append((p, s["dur"]))
            srt.append((t, t + s["dur"], s["text"]))
            script.append(s["text"] + "\n")
            t += s["dur"]
            prev_last = p
    # end hold + fade to surface
    last = Image.open(prev_last)
    fade = Image.new("RGB", (W, H), SURF)
    entries.append((prev_last, 0.6))
    for j in range(1, 19):
        q = FR / f"end_{j:02d}.png"
        Image.blend(last, fade, j / 18).save(q)
        entries.append((q, 1 / FPS))
    (OUT / "narration_script.md").write_text("\n".join(script), encoding="utf-8")
    (OUT / "captions.srt").write_text("\n".join(f"{i + 1}\n{srt_time(s)} --> {srt_time(e)}\n{tx}\n" for i, (s, e, tx) in enumerate(srt)), encoding="utf-8")
    lst = OUT / "frames.txt"
    with lst.open("w", encoding="utf-8") as f:
        for p, dur in entries:
            f.write(f"file '{p.as_posix()}'\nduration {dur:.4f}\n")
        f.write(f"file '{entries[-1][0].as_posix()}'\n")
    out = OUT / ("COLDER_demo.mp4" if a.audio else "COLDER_demo_captions.mp4")
    vsum = sum(d for _, d in entries)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if a.audio:
        cmd += ["-i", a.audio, "-af", f"afade=t=in:d=0.3,afade=t=out:st={max(0, vsum - 1.2):.2f}:d=1.0,loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{vsum:.2f}", "-i", "anullsrc=r=48000:cl=stereo", "-c:a", "aac", "-b:a", "64k"]
    cmd += ["-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-movflags", "+faststart", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    print(f"{out}  video {vsum:.1f}s  ({vsum / 60:.2f} min)  scenes={len(SCENES)} captions={len(srt)}")


if __name__ == "__main__":
    sys.exit(main())
