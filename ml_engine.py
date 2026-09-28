import re, json, copy
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION LEXICON
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_META = {
    "joy":      {"icon": "😊", "color": "#F59E0B", "keywords": ["happy","glad","excited","thrilled","delighted","wonderful","great","fantastic","awesome","love","joyful","pleased","overjoyed","elated","cheerful","ecstatic","grateful","thankful","blessed","amazing","brilliant","yay","hooray","celebrate","wonderful"]},
    "sadness":  {"icon": "😢", "color": "#3B82F6", "keywords": ["sad","unhappy","depressed","miserable","crying","cry","tears","grief","sorrowful","heartbroken","disappointed","devastated","upset","gloomy","melancholy","hopeless","lonely","lost","miss","hurt","pain","suffering","broken","gutted"]},
    "anger":    {"icon": "😠", "color": "#EF4444", "keywords": ["angry","furious","rage","mad","annoyed","irritated","frustrated","outraged","infuriated","hostile","hate","hatred","resent","bitter","livid","enraged","disgusted","disgust","fed up","sick of","betrayed","furious","livid","unfair"]},
    "fear":     {"icon": "😨", "color": "#8B5CF6", "keywords": ["afraid","scared","terrified","frightened","fearful","anxious","nervous","worried","panicked","horror","dread","trembling","shaking","threatened","unsafe","danger","nightmare","terrifying","panic","anxiety"]},
    "surprise": {"icon": "😲", "color": "#10B981", "keywords": ["surprised","shocked","astonished","amazed","stunned","unexpected","unbelievable","incredible","wow","omg","really","no way","seriously","sudden","suddenly","can't believe","never thought","startled"]},
    "disgust":  {"icon": "🤢", "color": "#6B7280", "keywords": ["disgusting","revolting","gross","nasty","awful","horrible","repulsive","despise","loathe","vile","repugnant","sick","yuck","shameful","appalling","revolting","hideous"]},
    "neutral":  {"icon": "😐", "color": "#94A3B8", "keywords": ["okay","fine","alright","sure","ok","noted","understood","i see","right","yes","no","maybe","perhaps"]},
}

LABELS = ["joy","sadness","anger","fear","surprise","disgust","neutral"]
CAUSAL_WORDS = ["because","since","due to","owing to","as a result","thanks to","caused by",
                "made me","makes me","that's why","so","therefore","hence","thus","after",
                "when","whenever","once","the reason","because of","resulted in"]
NEGATION_WORDS = {
    "not","no","never","none","neither","nor",
    "don't","doesn't","didn't","won't","isn't","aren't",
    "wasn't","weren't","can't","couldn't","wouldn't","shouldn't",
    "cannot","cannot","haven't","hasn't","hadn't","needn't","daren't",
    "hardly","barely","scarcely","seldom","rarely","nothing","nobody","nowhere",
    # contractions without apostrophe (after tokenization)
    "dont","doesnt","didnt","wont","isnt","arent",
    "wasnt","werent","cant","couldnt","wouldnt","shouldnt",
    "havent","hasnt","hadnt",
}

# Maps each emotion to the one activated when that emotion's keywords are negated
# e.g. "not happy" → sadness, "not angry" → neutral/calm
NEGATION_FLIP = {
    "joy":      "sadness",
    "sadness":  "neutral",
    "anger":    "neutral",
    "fear":     "neutral",
    "surprise": "neutral",
    "disgust":  "neutral",
}

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN LABELLED CORPUS  (~200 samples, 7 emotions)
# ─────────────────────────────────────────────────────────────────────────────
BUILTIN_CORPUS = [
    # joy
    ("I am so happy today, everything is going wonderfully!", "joy"),
    ("This is the best news I have ever received in my life.", "joy"),
    ("I feel absolutely thrilled and grateful for this opportunity.", "joy"),
    ("We won! I am overjoyed and cannot stop smiling!", "joy"),
    ("Thank you so much, you have made my day completely.", "joy"),
    ("I love this so much, it makes me feel wonderful.", "joy"),
    ("This is amazing, I am delighted beyond words.", "joy"),
    ("I am ecstatic about the results, truly blessed.", "joy"),
    ("You did it! I am so proud and happy for you.", "joy"),
    ("What a fantastic surprise, I feel so cheerful now!", "joy"),
    ("I got the promotion! I am elated and so excited.", "joy"),
    ("Finally some good news. I feel relief and joy.", "joy"),
    ("I am celebrating today because everything worked out perfectly.", "joy"),
    ("Your kind words make me feel so joyful and loved.", "joy"),
    ("I could not be happier about this outcome.", "joy"),
    ("Best day ever! I am overjoyed and grateful.", "joy"),
    ("We are finally together again, I feel so glad.", "joy"),
    ("This achievement fills me with immense happiness.", "joy"),
    ("I feel blessed and lucky every single day.", "joy"),
    ("Everything is perfect, I am beaming with joy.", "joy"),
    # sadness
    ("I feel so sad and lonely, nobody understands me.", "sadness"),
    ("I am devastated by the news, I cannot stop crying.", "sadness"),
    ("This heartbreak is unbearable, I am completely broken.", "sadness"),
    ("I miss you so much it hurts every single day.", "sadness"),
    ("I am deeply disappointed in what happened.", "sadness"),
    ("I feel hopeless and empty inside.", "sadness"),
    ("Losing someone you love leaves a pain that never fades.", "sadness"),
    ("I am grieving and it is so hard to move on.", "sadness"),
    ("Nothing feels right anymore, I am so unhappy.", "sadness"),
    ("I cried all night because of what you said.", "sadness"),
    ("I feel miserable and cannot find any joy.", "sadness"),
    ("The sadness is overwhelming and I cannot shake it.", "sadness"),
    ("I am melancholy and it is affecting everything I do.", "sadness"),
    ("I feel so upset and hurt by this situation.", "sadness"),
    ("My heart is broken and I feel lost.", "sadness"),
    ("I am suffering and do not know how to cope.", "sadness"),
    ("Everything reminds me of what I lost and I feel sad.", "sadness"),
    ("I feel like I am drowning in sorrow.", "sadness"),
    ("The pain of this loss is unbearable for me.", "sadness"),
    ("I am gutted and deeply sorrowful about this.", "sadness"),
    # anger
    ("I am absolutely furious about what just happened.", "anger"),
    ("This is completely unfair and I am outraged!", "anger"),
    ("How dare you do this to me, I am livid!", "anger"),
    ("I hate this situation, it makes me so angry.", "anger"),
    ("I am fed up with being treated this way.", "anger"),
    ("I cannot stand the injustice, it enrages me.", "anger"),
    ("You betrayed me and I am beyond furious.", "anger"),
    ("This infuriates me to no end, I am so mad.", "anger"),
    ("I resent how I have been treated here.", "anger"),
    ("Your lies make me bitter and angry.", "anger"),
    ("I am disgusted by the way this was handled.", "anger"),
    ("Stop dismissing me, it makes me furious.", "anger"),
    ("I am so irritated and annoyed by your behaviour.", "anger"),
    ("This is wrong and I am hostile towards it.", "anger"),
    ("You had no right to do that and I am enraged.", "anger"),
    ("I feel nothing but rage at this betrayal.", "anger"),
    ("This system is broken and I am furious about it.", "anger"),
    ("I am angry because nobody listened to me.", "anger"),
    ("My patience is gone and I am deeply frustrated.", "anger"),
    ("How could you, I am absolutely livid right now.", "anger"),
    # fear
    ("I am so scared and do not know what will happen.", "fear"),
    ("The thought of it terrifies me completely.", "fear"),
    ("I feel anxious and cannot calm down at all.", "fear"),
    ("I am afraid of what the future might bring.", "fear"),
    ("This situation is making me panic badly.", "fear"),
    ("I am worried that something terrible will happen.", "fear"),
    ("I feel nervous and my hands are shaking.", "fear"),
    ("The nightmare keeps repeating and I am terrified.", "fear"),
    ("I dread every morning because of the anxiety.", "fear"),
    ("I feel unsafe and frightened all the time.", "fear"),
    ("The horror of what could happen keeps me up at night.", "fear"),
    ("I am trembling with fear about the diagnosis.", "fear"),
    ("My anxiety is through the roof right now.", "fear"),
    ("I am petrified about starting this new chapter.", "fear"),
    ("The thought of failure makes me fearful.", "fear"),
    ("I feel threatened and scared in this environment.", "fear"),
    ("I cannot shake this overwhelming sense of dread.", "fear"),
    ("I am terrified of losing everything I worked for.", "fear"),
    ("This uncertainty fills me with deep fear.", "fear"),
    ("I feel paralysed by anxiety and cannot act.", "fear"),
    # surprise
    ("I cannot believe this happened, I am completely shocked!", "surprise"),
    ("Wow, I never expected this at all!", "surprise"),
    ("This is so unexpected, I am totally astonished.", "surprise"),
    ("No way! I am utterly stunned by this news.", "surprise"),
    ("I am amazed, I did not see this coming.", "surprise"),
    ("This took me by complete surprise, unbelievable!", "surprise"),
    ("OMG I cannot believe you did this for me!", "surprise"),
    ("I am speechless, this is absolutely stunning.", "surprise"),
    ("What a plot twist, I am genuinely surprised.", "surprise"),
    ("I never thought this would happen, I am shocked.", "surprise"),
    ("This is incredible, I am thoroughly astonished.", "surprise"),
    ("Wait, seriously? I am completely taken aback.", "surprise"),
    ("I cannot wrap my head around this surprise.", "surprise"),
    ("Nobody saw this coming, we are all shocked.", "surprise"),
    ("This revelation left me stunned and speechless.", "surprise"),
    ("I was not expecting this at all, what a shock.", "surprise"),
    ("The surprise was so big I burst into tears.", "surprise"),
    ("I am amazed at how this turned out.", "surprise"),
    ("Totally out of the blue and I am stunned.", "surprise"),
    ("I am so surprised and honestly a bit overwhelmed.", "surprise"),
    # disgust
    ("I am disgusted by this behaviour, it is revolting.", "disgust"),
    ("This is absolutely appalling and makes me sick.", "disgust"),
    ("I find this completely revolting and unacceptable.", "disgust"),
    ("The way they acted is disgusting and shameful.", "disgust"),
    ("I loathe this situation, it is utterly repulsive.", "disgust"),
    ("How can anyone do something so horrible and vile.", "disgust"),
    ("This makes me feel sick to my stomach.", "disgust"),
    ("I despise what you have become, it is gross.", "disgust"),
    ("This is morally repugnant and I am disgusted.", "disgust"),
    ("I cannot stand the sight of this, it is nasty.", "disgust"),
    ("The sheer audacity is revolting and disgusting.", "disgust"),
    ("I feel nothing but disgust and contempt for this.", "disgust"),
    ("This is shameful, awful and I am deeply disgusted.", "disgust"),
    ("What a vile and repulsive thing to say or do.", "disgust"),
    ("I am repelled by everything about this situation.", "disgust"),
    ("This is hideous and utterly unacceptable to me.", "disgust"),
    ("The corruption here is disgusting and sickening.", "disgust"),
    ("I cannot believe how gross and awful this is.", "disgust"),
    ("This rotten behaviour makes me sick with disgust.", "disgust"),
    ("I am horrified and disgusted in equal measure.", "disgust"),
    # neutral
    ("I will check the report and get back to you.", "neutral"),
    ("Okay, I understand what you are saying.", "neutral"),
    ("Let me know when you are ready to proceed.", "neutral"),
    ("Sure, I can look into that for you.", "neutral"),
    ("Alright, noted. I will follow up tomorrow.", "neutral"),
    ("I see, that makes sense given the context.", "neutral"),
    ("Yes, we can discuss this further later.", "neutral"),
    ("Fine, whatever works best for the team.", "neutral"),
    ("I will send you the document by end of day.", "neutral"),
    ("The meeting has been rescheduled to Monday.", "neutral"),
    ("Please let me know if you need anything else.", "neutral"),
    ("The report is ready and I have reviewed it.", "neutral"),
    ("We will proceed as discussed in the meeting.", "neutral"),
    ("I have received your message and will respond.", "neutral"),
    ("The task is complete, everything is in order.", "neutral"),
    ("No issues to report, everything is running fine.", "neutral"),
    ("I understand, I will take note of that.", "neutral"),
    ("The schedule remains the same for now.", "neutral"),
    ("We can revisit this at the next review.", "neutral"),
    ("The information has been recorded accordingly.", "neutral"),
    # ── negation examples ────────────────────────────────────────────────────
    # negated joy → sadness
    ("I am not happy about this at all.", "sadness"),
    ("I am not happy because my exam was not good.", "sadness"),
    ("This does not make me happy, it makes me miserable.", "sadness"),
    ("I don't feel joyful, I feel completely empty inside.", "sadness"),
    ("I'm not glad this happened, I'm heartbroken.", "sadness"),
    ("She is not excited about anything anymore.", "sadness"),
    ("I don't feel cheerful today, I feel really down.", "sadness"),
    ("Nothing about this is wonderful, I feel awful.", "sadness"),
    ("I am not thrilled, I am deeply disappointed.", "sadness"),
    ("It's not a great day, everything went wrong.", "sadness"),
    # negated joy → anger
    ("I'm not happy with how they treated me, I'm furious.", "anger"),
    ("I don't feel pleased at all, I'm outraged by this.", "anger"),
    ("This is not something I'm glad about, it infuriates me.", "anger"),
    # negated sadness → neutral / relief
    ("I'm not sad anymore, things are looking up.", "neutral"),
    ("She doesn't feel miserable today, she seems fine.", "neutral"),
    ("I no longer feel hopeless, I've moved on.", "neutral"),
    # negated anger → neutral
    ("I'm not angry anymore, I've calmed down.", "neutral"),
    ("He doesn't seem furious, he took it pretty well.", "neutral"),
    ("I wasn't mad about it, just a bit surprised.", "surprise"),
    # negated fear → neutral / joy
    ("I'm not scared at all, I feel perfectly safe.", "neutral"),
    ("She isn't worried anymore, the results were clear.", "neutral"),
    ("I don't dread it now, I'm actually looking forward to it.", "joy"),
    # negated disgust → neutral
    ("I don't find it revolting, it's actually fine.", "neutral"),
    ("He wasn't disgusted by it, he accepted it calmly.", "neutral"),
    # double negation (e.g. "not unhappy" → joy/neutral)
    ("I'm not unhappy about it, actually it went okay.", "neutral"),
    ("She is not displeased, she seems fairly content.", "neutral"),
    # general negated-positive → sadness
    ("I don't feel good about this situation at all.", "sadness"),
    ("This isn't making me feel any better.", "sadness"),
    ("I can't say I'm enjoying this, it's quite painful.", "sadness"),
    ("None of this feels right, I'm very upset.", "sadness"),
    ("I never feel happy when things like this happen.", "sadness"),
    ("It's not okay and I don't feel fine.", "sadness"),
    ("I haven't felt joy in a very long time.", "sadness"),
    ("I don't feel satisfied with how things turned out.", "sadness"),
    # negated-positive with cause
    ("I am not happy because my exam was not good.", "sadness"),
    ("I don't feel great because the news was terrible.", "sadness"),
    ("I can't feel excited since everything went wrong.", "sadness"),
    ("I won't be glad until this situation is fixed.", "sadness"),
    ("I'm not okay because nobody listened to me.", "anger"),
    # ── augmented multi-sentence ──────────────────────────────────────────────
    ("After hearing the terrible news, I feel deeply sad and broken.", "sadness"),
    ("Because you lied to me, I am furious and cannot trust you.", "anger"),
    ("Since you got that promotion, I have been so happy for you.", "joy"),
    ("When I heard the loud noise, I was terrified and froze.", "fear"),
    ("I never thought they would do this, I am completely shocked.", "surprise"),
    ("The way they treated others is revolting and I am disgusted.", "disgust"),
    ("I was doing fine until you brought that up again.", "neutral"),
    ("This wonderful moment makes my heart overflow with joy.", "joy"),
    ("I am shattered because of the way things ended.", "sadness"),
    ("Your constant disrespect makes me livid beyond words.", "anger"),
    ("I dread every phone call in case it is bad news.", "fear"),
    ("Wow no one expected that twist, it is stunning.", "surprise"),
    ("This behaviour is absolutely repugnant, I am sickened.", "disgust"),
    ("Everything seems normal and I have no strong feelings.", "neutral"),
    ("I feel overjoyed because my family is back together.", "joy"),
    ("I am heartbroken that things could not work out.", "sadness"),
    ("I am so frustrated because nobody takes me seriously.", "anger"),
    ("I am anxious about the results and cannot sleep.", "fear"),
    ("I am astonished that they managed to pull this off.", "surprise"),
    ("This disgusting lack of accountability makes me ill.", "disgust"),
    ("The situation is as expected, nothing out of the ordinary.", "neutral"),
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def _has_negation(tokens: List[str], idx: int, win=3) -> bool:
    return any(t in NEGATION_WORDS for t in tokens[max(0, idx-win):idx])

# Regex that matches a negation word followed by up to 2 filler words then an
# emotion keyword.  The matched span is replaced with "NEG_<emotion>" so that
# TF-IDF learns the negated token as a distinct feature.
#   e.g. "not very happy"  → "NEG_joy"
#        "don't feel angry" → "NEG_anger"
_NEG_PATTERN = re.compile(
    r"\b(?:" +
    "|".join(re.escape(n) for n in sorted(NEGATION_WORDS, key=len, reverse=True)) +
    r")\b(?:\s+\w+){0,3}?\s+(?P<kw>\w+)",
    re.IGNORECASE,
)

def _preprocess_negation(text: str) -> str:
    """
    Rewrite negated emotion keywords so TF-IDF treats them as distinct tokens.

    'i am not happy'        → 'i am NEG_joy'
    'i don't feel angry'    → 'i NEG_anger'
    'she isn't scared'      → 'she NEG_fear'

    Only rewrites when the keyword following the negation actually belongs to
    an emotion's lexicon.  Leaves all other text unchanged.
    """
    # Build a flat keyword→emotion lookup once (O(1) per call after that)
    kw_to_emo: Dict[str, str] = {}
    for emo, meta in EMOTION_META.items():
        for kw in meta["keywords"]:
            # only single-word keywords are caught by token-level regex
            if " " not in kw:
                kw_to_emo[kw.lower()] = emo

    def _replace(m: re.Match) -> str:
        kw = m.group("kw").lower()
        if kw in kw_to_emo:
            emo = kw_to_emo[kw]
            # Replace the entire matched span with the NEG token
            return f"NEG_{emo}"
        # keyword not in any lexicon — leave the match as-is
        return m.group(0)

    return _NEG_PATTERN.sub(_replace, text)

def lexicon_feature_vector(text: str) -> List[float]:
    """
    15-dim hand-crafted feature vector.

    Negation handling:
      - When a negation word appears within a 3-token window BEFORE an emotion
        keyword, the hit for that emotion is zeroed out and instead credited to
        the NEGATION_FLIP target emotion (e.g. "not happy" → sadness +1, joy +0).
      - A small residual (0.15) is kept on the original emotion so the model
        can still distinguish "not happy" from a fully unrelated sentence.
    """
    tokens = _tokenize(text)
    # accumulate raw scores per emotion label (excluding neutral for now)
    emo_scores = {emo: 0.0 for emo in LABELS}

    for emo in LABELS[:-1]:  # skip neutral
        for i, t in enumerate(tokens):
            if t in EMOTION_META[emo]["keywords"]:
                negated = _has_negation(tokens, i)
                if negated:
                    # suppress the matched emotion, credit the flipped target
                    flip_target = NEGATION_FLIP.get(emo, "neutral")
                    emo_scores[emo]         += 0.15   # small residual
                    emo_scores[flip_target] += 0.85   # main credit goes to flip
                else:
                    emo_scores[emo] += 1.0

    scores = [emo_scores[e] for e in LABELS[:-1]]   # 6 values, neutral excluded

    has_causal  = float(any(cw in text.lower() for cw in CAUSAL_WORDS))
    has_neg     = float(any(n in tokens for n in NEGATION_WORDS))
    text_len    = min(len(tokens) / 30.0, 1.0)
    exclaim     = float("!" in text)
    question    = float("?" in text)
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    any_hit     = float(sum(scores) > 0)
    neg_joy     = float(has_neg and emo_scores["sadness"] > emo_scores["joy"])  # extra signal

    return scores + [has_causal, has_neg, text_len, exclaim, question, upper_ratio, any_hit, neg_joy]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (factory so we can get fresh instances for retraining)
# ─────────────────────────────────────────────────────────────────────────────
def _fresh_models() -> Dict:
    return {
        "Logistic Regression": LogisticRegression( max_iter=1000, C=1.0, solver="lbfgs"),
        "SVM":                 LinearSVC(max_iter=2000, C=1.0),
        "Naive Bayes":         MultinomialNB(alpha=0.5),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ML ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EmotionMLEngine:
    def __init__(self):
        self.vectorizer   = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        self.le           = LabelEncoder()
        self.trained      = {}   # model_name -> fitted model
        self.metrics      = {}   # model_name -> metric dict
        self.active_model = "Random Forest"
        self.training_size = len(BUILTIN_CORPUS)
        # snapshot of metrics before the most recent retrain (for comparison)
        self.prev_metrics: Dict = {}
        self._train(BUILTIN_CORPUS)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _build_features(self, texts):
        import scipy.sparse as sp
        preprocessed = [_preprocess_negation(t) for t in texts]
        tfidf = self.vectorizer.transform(preprocessed)
        lex   = np.array([lexicon_feature_vector(t) for t in texts])
        return sp.hstack([tfidf, sp.csr_matrix(lex)])

    def _train(self, corpus: List[Tuple[str, str]]):
        """Train (or retrain) all four models on the given corpus."""
        texts  = [c[0] for c in corpus]
        labels = [c[1] for c in corpus]

        self.le.fit(LABELS)          # always fit on full label set so classes are stable
        y = self.le.transform(labels)

        preprocessed_texts = [_preprocess_negation(t) for t in texts]
        self.vectorizer.fit(preprocessed_texts)
        X = self._build_features(texts)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42,
            stratify=y if len(set(labels)) > 1 else None
        )

        new_metrics = {}
        new_trained = {}
        models = _fresh_models()

        for name, model in models.items():
            if name == "Naive Bayes":
                tfidf_all = self.vectorizer.transform(preprocessed_texts)
                from sklearn.model_selection import train_test_split as tts
                Xn_tr, Xn_te, yn_tr, yn_te = tts(
                    tfidf_all, y, test_size=0.25, random_state=42,
                    stratify=y if len(set(labels)) > 1 else None
                )
                model.fit(Xn_tr, yn_tr)
                y_pred     = model.predict(Xn_te)
                y_te_use   = yn_te
                cv_X       = tfidf_all
            else:
                model.fit(X_tr, y_tr)
                y_pred   = model.predict(X_te)
                y_te_use = y_te
                cv_X     = X

            acc  = round(accuracy_score(y_te_use, y_pred), 4)
            prec = round(precision_score(y_te_use, y_pred, average="weighted", zero_division=0), 4)
            rec  = round(recall_score(y_te_use, y_pred, average="weighted", zero_division=0), 4)
            f1   = round(f1_score(y_te_use, y_pred, average="weighted", zero_division=0), 4)
            cm   = confusion_matrix(
                y_te_use, y_pred,
                labels=list(range(len(self.le.classes_)))
            ).tolist()

            try:
                cv = cross_val_score(model, cv_X, y, cv=5, scoring="f1_weighted")
                cv_mean = round(float(cv.mean()), 4)
                cv_std  = round(float(cv.std()),  4)
            except Exception:
                cv_mean, cv_std = f1, 0.0

            new_metrics[name] = {
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1":        f1,
                "cv_f1_mean": cv_mean,
                "cv_f1_std":  cv_std,
                "confusion_matrix": cm,
                "labels": list(self.le.classes_),
            }
            new_trained[name] = model

        self.metrics = new_metrics
        self.trained = new_trained
        self.training_size = len(corpus)

    # ── Public: retrain with extra examples from DB ───────────────────────────
    def retrain(self, extra_examples: List[Dict]) -> Dict:
        """
        Retrain all models on BUILTIN_CORPUS + extra_examples.
        extra_examples: list of {"text": str, "emotion": str}
        Returns {"before": metrics_snapshot, "after": new_metrics, "delta": diff}
        """
        # snapshot before retrain
        self.prev_metrics = copy.deepcopy(self.metrics)

        combined = list(BUILTIN_CORPUS) + [(e["text"], e["emotion"]) for e in extra_examples]
        # filter to valid labels only
        combined = [(t, l) for t, l in combined if l in LABELS]

        self._train(combined)

        delta = {}
        for name in self.metrics:
            before = self.prev_metrics.get(name, {})
            after  = self.metrics[name]
            delta[name] = {
                "accuracy_delta":  round(after["accuracy"]  - before.get("accuracy", 0),  4),
                "precision_delta": round(after["precision"] - before.get("precision", 0), 4),
                "recall_delta":    round(after["recall"]    - before.get("recall", 0),    4),
                "f1_delta":        round(after["f1"]        - before.get("f1", 0),        4),
            }

        return {
            "before":        self.prev_metrics,
            "after":         self.metrics,
            "delta":         delta,
            "training_size": self.training_size,
        }

    # ── Get comparison snapshot (before vs after) ─────────────────────────────
    def get_comparison_snapshot(self) -> Dict:
        """Returns before/after comparison if a retrain has happened."""
        if not self.prev_metrics:
            return {}
        delta = {}
        for name in self.metrics:
            before = self.prev_metrics.get(name, {})
            after  = self.metrics[name]
            delta[name] = {
                "accuracy_delta":  round(after["accuracy"]  - before.get("accuracy", 0),  4),
                "f1_delta":        round(after["f1"]        - before.get("f1", 0),        4),
            }
        return {
            "before": self.prev_metrics,
            "after":  self.metrics,
            "delta":  delta,
            "training_size": self.training_size,
        }

    # ── Predict single text ────────────────────────────────────────────────────
    def predict(self, text: str, model_name: Optional[str] = None) -> Dict:
        name  = model_name or self.active_model
        model = self.trained[name]

        if name == "Naive Bayes":
            X = self.vectorizer.transform([_preprocess_negation(text)])
        else:
            X = self._build_features([text])

        pred_idx = model.predict(X)[0]
        emotion  = self.le.inverse_transform([pred_idx])[0]

        confidence = 0.75
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = round(float(probs.max()), 4)
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)[0]
                if df.ndim == 0:
                    df = np.array([df])
                exp   = np.exp(df - df.max())
                probs = exp / exp.sum()
                confidence = round(float(probs.max()), 4)
        except Exception:
            pass

        return {
            "emotion":    emotion,
            "confidence": confidence,
            "icon":       EMOTION_META[emotion]["icon"],
            "color":      EMOTION_META[emotion]["color"],
            "model":      name,
        }

    # ── Predict with all models ────────────────────────────────────────────────
    def predict_all(self, text: str) -> Dict:
        results = {}
        for name in self.trained:
            results[name] = self.predict(text, name)
        votes  = Counter(r["emotion"] for r in results.values())
        winner = votes.most_common(1)[0][0]
        return {"per_model": results, "ensemble": winner, "vote_counts": dict(votes)}

    # ── Analyse full conversation ──────────────────────────────────────────────
    def analyse_conversation(self, utterances: List[Dict], model_name: Optional[str] = None) -> Dict:
        annotated = []
        for u in utterances:
            pred = self.predict(u["text"], model_name)
            annotated.append({**u, **pred})

        pairs = self._extract_cause_pairs(annotated)

        trend = [{"turn": a["turn_index"] + 1, "speaker": a["speaker"],
                  "emotion": a["emotion"], "color": a["color"], "icon": a["icon"],
                  "confidence": a["confidence"]} for a in annotated]

        dist = Counter(a["emotion"] for a in annotated)

        return {
            "utterances": annotated,
            "pairs":      pairs,
            "trend":      trend,
            "stats": {
                "total_utterances":    len(utterances),
                "total_pairs":         len(pairs),
                "emotion_distribution": dict(dist),
                "dominant_emotion":    dist.most_common(1)[0][0] if dist else "neutral",
                "model_used":          model_name or self.active_model,
            },
        }

    # ── Cause pair extraction ──────────────────────────────────────────────────
    def _cause_score(self, emo_utt: Dict, cand: Dict) -> Tuple[float, str]:
        score = 0.0
        reasons = []
        dist = emo_utt["turn_index"] - cand["turn_index"]
        if dist > 0:
            score += 0.4 / dist
        if any(cw in cand["text"].lower() for cw in CAUSAL_WORDS):
            score += 0.25; reasons.append("causal connective")
        e_tok = set(_tokenize(emo_utt["text"])) - NEGATION_WORDS - {"i","you","the","a","is","was","it"}
        c_tok = set(_tokenize(cand["text"]))     - NEGATION_WORDS - {"i","you","the","a","is","was","it"}
        ov = len(e_tok & c_tok)
        if ov > 0:
            score += min(ov * 0.08, 0.25); reasons.append(f"{ov} shared words")
        if cand["speaker"] != emo_utt["speaker"]:
            score += 0.08; reasons.append("cross-speaker trigger")
        if "?" in cand["text"] or cand["text"].lower().startswith(("why","what","how","did you","do you","are you")):
            score += 0.07; reasons.append("interrogative turn")
        return round(min(score, 0.99), 3), "; ".join(reasons) or "proximity"

    def _extract_cause_span(self, text: str) -> str:
        for cw in CAUSAL_WORDS:
            idx = text.lower().find(cw)
            if idx != -1:
                span = text[idx + len(cw):].strip().rstrip(".,!?")
                if len(span) > 3:
                    return span[:120]
        return text[:100]

    def _extract_cause_pairs(self, annotated: List[Dict], window=6) -> List[Dict]:
        pairs = []
        for i, utt in enumerate(annotated):
            if utt["emotion"] == "neutral":
                continue
            window_cands = annotated[max(0, i - window): i + 1]
            best_score, best_cand, best_exp = -1.0, None, ""
            for cand in window_cands:
                sc, exp = self._cause_score(utt, cand)
                if sc > best_score:
                    best_score, best_cand, best_exp = sc, cand, exp
            if best_cand and best_score >= 0.1:
                pairs.append({
                    "emotion_turn":         utt["turn_index"],
                    "cause_turn":           best_cand["turn_index"],
                    "emotion":              utt["emotion"],
                    "emotion_text":         utt["text"],
                    "cause_text":           best_cand["text"],
                    "emotion_speaker":      utt["speaker"],
                    "cause_speaker":        best_cand["speaker"],
                    "cause_span":           self._extract_cause_span(best_cand["text"]),
                    "confidence":           best_score,
                    "explanation":          best_exp,
                    "emotion_utterance_id": utt.get("utterance_id"),
                    "cause_utterance_id":   best_cand.get("utterance_id"),
                    "icon":                 utt["icon"],
                    "color":                utt["color"],
                })
        return pairs

    # ── Metrics summaries ──────────────────────────────────────────────────────
    def get_all_metrics(self) -> Dict:
        return self.metrics

    def get_comparison_table(self) -> List[Dict]:
        rows = []
        for name, m in self.metrics.items():
            rows.append({
                "model":     name,
                "accuracy":  m["accuracy"],
                "precision": m["precision"],
                "recall":    m["recall"],
                "f1":        m["f1"],
                "cv_f1":     m["cv_f1_mean"],
                "cv_std":    m["cv_f1_std"],
            })
        return sorted(rows, key=lambda x: x["f1"], reverse=True)


# Singleton — loaded once at app startup
engine = EmotionMLEngine()
