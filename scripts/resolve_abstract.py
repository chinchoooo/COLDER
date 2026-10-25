"""Resolve the LaTeX abstract's number macros into plain text for the Apart submission form (results/abstract.txt)."""
import re

tex = open("paper/main.tex", encoding="utf-8").read()
nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open("paper/numbers.tex", encoding="utf-8").read()))
ab = re.search(r"\\begin\{abstract\}\\small\\noindent\s*(.*?)\\end\{abstract\}", tex, re.S).group(1)
ab = re.sub(r"\\(\w+)\{\}", lambda m: nums[m.group(1)], ab)
BS = chr(92)
ab = ab.replace("~", " ").replace("--", "\u2013").replace(BS + " ", " ").replace("``", "\u201c").replace("''", "\u201d").replace(BS, "")
ab = re.sub(r"\s+", " ", ab).strip()
open("results/abstract.txt", "w", encoding="utf-8").write(ab)
print(len(ab.split()), "words")
print(ab)
