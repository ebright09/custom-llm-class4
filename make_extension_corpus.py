"""Write my extension teaching corpus into corpus/extension/*.md.

Targets four extension-eval categories: opposites, negation, spatial relations,
and categories/analogies. The text is synthetic and written for this project.

Separation rules I follow (stricter than the notebook's exact-prefix check):
* No eval prompt appears anywhere (checked below with the course's own matcher).
* The exact word pairs that eval items ask about are never taught in the tested
  relation. Example: hot/cold, empty/full and noisy/quiet never appear in an
  "opposite" or contrast frame; they only appear in neutral descriptive sentences
  so they are in the vocabulary. The same goes for book-in-bag, lamp-above-desk,
  ball-left-of-box, box red/blue, ava tea/milk, door open/closed, robin/salmon,
  puppy/dog + kitten/cat and carrot/apple.
* No names used in the eval suite are used here.

Formatting note: the notebook's chunker splits a passage after every ". " (period
followed by whitespace). Multi-sentence examples are therefore written without the
space ("not red.it is blue.") so they stay one passage. The tokenizer produces
exactly the same tokens ("red", ".", "it") as the spaced version.

Revision 2: the v1 run showed that a case is scorable only when every one of its
four answer choices is in the vocabulary. v1 taught the answers but not several
distractor words, so revision 2 adds those ordinary words (heavy, early, loud,
round, late, missing, beside, north, south) in neutral sentences that do not use
the tested relation. Eval names (e.g. ava) are still never added. Run with
--v1 to regenerate the first version.

    python make_extension_corpus.py
"""
import sys
import itertools
import random
from pathlib import Path

from run_evals import load_suite, matching_cases

rng = random.Random(7)
OUT = Path("corpus/extension")


def sample(items, k):
    items = sorted(set(items))
    return rng.sample(items, min(k, len(items)))


def join(*sentences):
    """One passage made of several sentences, kept together (see module docstring)."""
    return ".".join(s.strip() for s in sentences) + "."


# ---------------------------------------------------------------- opposites
PAIRS = [("big", "small"), ("tall", "short"), ("fast", "slow"), ("happy", "sad"),
         ("rich", "poor"), ("young", "old"), ("thick", "thin"), ("strong", "weak"),
         ("near", "far"), ("clean", "dirty"), ("easy", "difficult"), ("high", "low"),
         ("wide", "narrow"), ("deep", "shallow"), ("sweet", "sour"), ("day", "night"),
         ("win", "lose"), ("begin", "end"), ("good", "bad"), ("true", "false"),
         ("hard", "soft"), ("up", "down"), ("smooth", "rough"), ("brave", "afraid"),
         ("sharp", "dull"), ("cheap", "expensive"), ("tidy", "messy"), ("bright", "dim")]
NOUNS = ["river", "road", "house", "tree", "story", "room", "shirt", "path", "rope", "coat",
         "garden", "window", "song", "town", "table", "cake"]
FRAMES = ["the opposite of {a} is {b} .", "{b} is the opposite of {a} .",
          "{a} and {b} are opposites .", "my teacher said the opposite of {a} is {b} .",
          "in class we learned that the opposite of {a} is {b} .",
          "a word that means the opposite of {a} is {b} ."]
NOT_ADJECTIVES = {("day", "night"), ("win", "lose"), ("begin", "end"), ("up", "down")}
opposites = []
for a, b in PAIRS:
    for x, y in [(a, b), (b, a)]:
        opposites += [f.format(a=x, b=y) for f in FRAMES]
        if (a, b) not in NOT_ADJECTIVES:
            opposites += [f"the first {n} was {x} , but the second {n} was {y} ." for n in sample(NOUNS, 4)]
        opposites.append(join(f"the {rng.choice(NOUNS)} is {x}", f"the opposite of {x} is {y}"))
# Neutral sentences so the tested words are known without teaching their opposites.
for word, things in {"hot": ["soup", "tea", "stove", "sand", "summer day"],
                     "cold": ["winter night", "river", "wind", "snow", "milk"],
                     "empty": ["glass", "room", "street", "jar", "basket"],
                     "full": ["bus", "cup", "theater", "basket", "train"],
                     "noisy": ["market", "street", "classroom", "party", "station"],
                     "quiet": ["library", "forest", "morning", "garden", "hall"]}.items():
    for thing in things:
        opposites += [f"the {thing} was {word} today .", f"we noticed that the {thing} is {word} ."]

# ----------------------------------------------------------------- negation
OBJECTS = ["cup", "car", "shirt", "hat", "chair", "kite", "bike", "wall", "flower", "coat",
           "ribbon", "pencil", "towel", "bottle", "umbrella", "house"]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "purple", "brown", "pink", "gray"]
negation = []
for obj in OBJECTS:
    for c1, c2 in itertools.permutations(COLORS, 2):
        negation.append(join(f"the {obj} is not {c1}", f"it is {c2}", f"the {obj} is {c2}"))
for obj in ["window", "gate", "shop", "drawer", "book", "laptop"]:
    for s1, s2 in [("open", "closed"), ("closed", "open")]:
        negation.append(join(f"the {obj} is not {s1}", f"it is {s2}", f"the {obj} is {s2}"))
        negation.append(join(f"the {obj} was not {s1}", f"it was {s2}"))
NAMES = [("lina", "she"), ("sam", "he"), ("rosa", "she"), ("jonas", "he"), ("mia", "she"),
         ("ben", "he"), ("zoe", "she"), ("ivan", "he"), ("kira", "she"), ("tom", "he")]
FOODS = ["coffee", "juice", "bread", "soup", "cake", "rice", "water", "cheese", "tea", "milk",
         "eggs", "pasta"]
VERBS = [("buy", "bought"), ("choose", "chose"), ("order", "ordered"),
         ("want", "wanted"), ("bring", "brought")]
for name, pron in NAMES:
    for base, past in VERBS:
        for x, y in itertools.permutations(FOODS, 2):
            if {x, y} == {"tea", "milk"}:
                continue
            negation.append(join(f"{name} did not {base} {x}", f"{pron} {past} {y}", f"{name} {past} {y}"))
for obj in OBJECTS[:8]:
    for a, b in [("big", "small"), ("small", "big"), ("new", "old"), ("old", "new"), ("dirty", "clean")]:
        negation.append(join(f"the {obj} is not {a}", f"it is {b}", f"the {obj} is {b}"))

# -------------------------------------------------------- spatial relations
SMALL = ["pen", "key", "coin", "letter", "phone", "spoon", "ring", "toy", "shirt", "book",
         "apple", "ticket", "card", "sock"]
CONTAINERS = ["drawer", "basket", "jar", "bag", "pocket", "box", "suitcase", "bowl", "cupboard"]
spatial = []
for thing, box in itertools.product(SMALL, CONTAINERS):
    if (thing, box) == ("book", "bag"):
        continue
    spatial.append(join(f"the {thing} is inside the {box}", f"the {box} contains the {thing}"))
    spatial.append(join(f"the {box} contains the {thing}", f"the {thing} is inside the {box}"))
STACK = ["clock", "shelf", "picture", "table", "chair", "bed", "window", "rug", "cloud", "roof",
         "floor", "bridge", "river", "lamp", "desk", "mirror", "sofa", "plant"]
for a, b in itertools.permutations(STACK, 2):
    if {a, b} == {"lamp", "desk"}:
        continue
    spatial.append(join(f"the {a} is above the {b}", f"the {b} is below the {a}"))
    spatial.append(join(f"the {a} is below the {b}", f"the {b} is above the {a}"))
ROW = ["cup", "plate", "chair", "tree", "car", "house", "shop", "door", "bench", "lamp",
       "window", "bottle", "vase", "ball", "box"]
for a, b in itertools.permutations(ROW, 2):
    if {a, b} == {"ball", "box"}:
        continue
    spatial.append(join(f"the {a} is left of the {b}", f"the {b} is to the right of the {a}"))
    spatial.append(join(f"the {a} is right of the {b}", f"the {b} is to the left of the {a}"))
spatial = sample(spatial, 900)

# -------------------------------------------------- categories and analogies
KINDS = {"bird": ["sparrow", "eagle", "owl", "parrot", "pigeon", "crow", "swan", "penguin"],
         "fish": ["trout", "shark", "tuna", "cod", "carp", "goldfish", "herring"],
         "tree": ["oak", "pine", "maple", "birch", "willow"],
         "tool": ["hammer", "saw", "wrench", "drill", "shovel"],
         "vegetable": ["potato", "onion", "pepper", "lettuce", "bean", "pea", "cabbage"],
         "fruit": ["banana", "pear", "mango", "peach", "grape", "lemon", "cherry"],
         "metal": ["iron", "copper", "silver", "gold", "steel"],
         "vehicle": ["van", "tractor", "scooter", "truck", "ship"],
         "fabric": ["cotton", "wool", "silk", "linen"]}
GROW = [("calf", "cow"), ("lamb", "sheep"), ("foal", "horse"), ("chick", "hen"),
        ("duckling", "duck"), ("piglet", "pig"), ("cub", "bear"), ("tadpole", "frog"),
        ("caterpillar", "butterfly"), ("kid", "goat"), ("fawn", "deer"), ("gosling", "goose")]
def article(word):
    return "an" if word[0] in "aeiou" else "a"
members = [(m, k) for k, ms in KINDS.items() for m in ms]
categories = []
for m, k in members:
    categories += [f"{article(m)} {m} is {article(k)} {k} .", f"we know that {article(m)} {m} is {article(k)} {k} ."]
for (m1, k1), (m2, k2) in itertools.permutations(members, 2):
    categories.append(join(f"{article(m1)} {m1} is {article(k1)} {k1}", f"{article(m2)} {m2} is {article(k2)} {k2}"))
categories = sample(categories, 700)
for baby, adult in GROW:
    categories.append(f"{article(baby)} {baby} grows into {article(adult)} {adult} .")
    categories.append(f"when {article(baby)} {baby} gets older it becomes {article(adult)} {adult} .")
for (b1, a1), (b2, a2) in itertools.permutations(GROW, 2):
    categories.append(join(f"{article(b1)} {b1} grows into {article(a1)} {a1}",
                           f"{article(b2)} {b2} grows into {article(a2)} {a2}"))
# Shared everyday contexts (no category or growth statements for tested items).
for fish in ["trout", "cod", "salmon", "herring"]:
    categories += [f"{article(fish)} {fish} swims in cold water .", f"the {fish} has fins and scales ."]
for animal in ["kitten", "cat", "puppy", "dog", "lamb", "calf"]:
    categories += [f"the {animal} slept on the warm rug .", f"the {animal} played in the garden ."]
for food in ["carrot", "potato", "onion", "apple", "pear", "mango", "robin"]:
    if food == "robin":
        categories += ["a robin sings in the morning .", "the robin built a nest in the tree ."]
    else:
        categories += [f"we cut the {food} for lunch .", f"the {food} was fresh from the farm ."]

negation_extra = []
if "--v1" not in sys.argv:
    # Revision 2: neutral sentences for distractor words (see module docstring).
    THINGS = ["suitcase", "table", "stone", "bag", "clock", "coin", "plate", "drum", "bell", "letter"]
    for word, frames in {
        "heavy": ["the {t} was heavy .", "we could not lift the heavy {t} ."],
        "early": ["we arrived early to see the {t} .", "the {t} was delivered early ."],
        "late": ["the {t} was delivered late .", "we arrived late to see the {t} ."],
        "loud": ["the {t} made a loud sound .", "we heard a loud {t} ."],
        "round": ["the {t} is round .", "she found a round {t} ."]}.items():
        opposites += [f.format(t=t) for f in frames for t in THINGS]
    for t in THINGS + ["key", "ticket", "phone", "ring", "card"]:
        negation_extra += [f"the {t} was missing this morning .", f"we looked for the missing {t} ."]
    for a, b in itertools.permutations(["chair", "tree", "car", "bench", "house", "shop", "sofa", "clock"], 2):
        spatial.append(f"the {a} stood beside the {b} .")
    for town, place in itertools.permutations(["town", "farm", "lake", "forest", "village", "bridge"], 2):
        spatial += [f"the {town} is north of the {place} .", f"the {town} is south of the {place} ."]

files = {"opposites.md": opposites, "negation.md": sample(negation, 900) + negation_extra,
         "spatial_relations.md": spatial, "categories_and_analogies.md": categories}
suite = load_suite()
OUT.mkdir(parents=True, exist_ok=True)
for name, lines in files.items():
    lines = sorted(set(lines))
    rng.shuffle(lines)
    text = "\n".join(lines) + "\n"
    leaks = sorted({i for line in lines for i in matching_cases(line, suite)} | set(matching_cases(text, suite)))
    if leaks:
        raise SystemExit(f"{name}: contains eval prompts {leaks}")
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"{name}: {len(lines)} passages")
