import json, os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from database import init_db, get_db
from ml_engine import engine, EMOTION_META

app = Flask(__name__)
app.secret_key = "ecpe-ai-2026"

# ── startup ──────────────────────────────────────────────────────────────────
with app.app_context():
    init_db()
    db = get_db()
    db.execute("DELETE FROM model_metrics")
    for name, m in engine.get_all_metrics().items():
        db.execute(
            """INSERT INTO model_metrics
               (model_name,accuracy,precision,recall,f1,cv_f1_mean,cv_f1_std,confusion_matrix,labels)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (name, m["accuracy"], m["precision"], m["recall"], m["f1"],
             m["cv_f1_mean"], m["cv_f1_std"],
             json.dumps(m["confusion_matrix"]), json.dumps(m["labels"]))
        )
    db.commit()
    db.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    db    = get_db()
    convs = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
    total_pairs = db.execute("SELECT COUNT(*) FROM emotion_cause_pairs").fetchone()[0]
    total_runs  = db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
    total_utts  = db.execute("SELECT COUNT(*) FROM utterances").fetchone()[0]
    db.close()
    return render_template("index.html",
        conversations=convs,
        total_pairs=total_pairs,
        total_runs=total_runs,
        total_utts=total_utts,
        emotion_meta=EMOTION_META,
    )

@app.route("/conversation/<int:conv_id>")
def conversation_view(conv_id):
    db   = get_db()
    conv = db.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    if not conv:
        db.close(); return redirect(url_for("index"))
    utts = db.execute(
        "SELECT * FROM utterances WHERE conversation_id=? ORDER BY turn_index", (conv_id,)
    ).fetchall()
    pairs = db.execute(
        """SELECT ecp.*,
               eu.text AS emotion_text, eu.speaker AS emotion_speaker, eu.turn_index AS etn,
               cu.text AS cause_text,   cu.speaker AS cause_speaker,   cu.turn_index AS ctn
           FROM emotion_cause_pairs ecp
           JOIN utterances eu ON ecp.emotion_utterance_id=eu.id
           JOIN utterances cu ON ecp.cause_utterance_id=cu.id
           WHERE ecp.conversation_id=? ORDER BY eu.turn_index""",
        (conv_id,)
    ).fetchall()
    run = db.execute(
        "SELECT * FROM analysis_runs WHERE conversation_id=? ORDER BY ran_at DESC LIMIT 1",
        (conv_id,)
    ).fetchone()
    db.close()
    return render_template("conversation.html",
        conv=conv, utterances=utts, pairs=pairs, run=run,
        emotion_meta=EMOTION_META,
        model_names=list(engine.trained.keys()),
        active_model=engine.active_model,
    )

@app.route("/analytics")
def analytics():
    db = get_db()
    metrics_rows = db.execute("SELECT * FROM model_metrics ORDER BY f1 DESC").fetchall()
    emo_dist = [dict(r) for r in db.execute(
        "SELECT emotion, COUNT(*) as cnt FROM utterances GROUP BY emotion"
    ).fetchall()]
    db.close()
    comparison = engine.get_comparison_table()

    parsed_metrics = []
    for r in metrics_rows:
        row = dict(r)
        row["confusion_matrix"] = json.loads(row["confusion_matrix"]) if isinstance(row["confusion_matrix"], str) else row["confusion_matrix"]
        row["labels"]           = json.loads(row["labels"])           if isinstance(row["labels"], str)           else row["labels"]
        parsed_metrics.append(row)

    return render_template("analytics.html",
        metrics_rows=parsed_metrics,
        emo_dist=emo_dist,
        comparison=comparison,
        emotion_meta=EMOTION_META,
    )

@app.route("/history")
def history():
    db = get_db()
    runs = db.execute(
        """SELECT ar.*, c.title FROM analysis_runs ar
           JOIN conversations c ON ar.conversation_id=c.id
           ORDER BY ar.ran_at DESC"""
    ).fetchall()
    db.close()
    return render_template("history.html", runs=runs)

@app.route("/about")
def about():
    return render_template("about.html")

# ─────────────────────────────────────────────────────────────────────────────
# API — CONVERSATIONS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/conversations", methods=["GET"])
def api_list_convs():
    db   = get_db()
    rows = db.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/conversations", methods=["POST"])
def api_create_conv():
    data  = request.get_json(force=True)
    title = (data.get("title") or "Untitled").strip()
    db    = get_db()
    cur   = db.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    cid   = cur.lastrowid
    db.commit()
    row   = db.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    db.close()
    return jsonify(dict(row)), 201

@app.route("/api/conversations/<int:cid>", methods=["DELETE"])
def api_delete_conv(cid):
    db = get_db()
    db.execute("DELETE FROM conversations WHERE id=?", (cid,))
    db.commit(); db.close()
    return jsonify({"deleted": cid})

@app.route("/api/conversations/<int:cid>/rename", methods=["PUT"])
def api_rename_conv(cid):
    data  = request.get_json(force=True)
    title = (data.get("title") or "Untitled").strip()
    db    = get_db()
    db.execute("UPDATE conversations SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, cid))
    db.commit(); db.close()
    return jsonify({"ok": True})

# ─────────────────────────────────────────────────────────────────────────────
# API — UTTERANCES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/conversations/<int:cid>/utterances", methods=["GET"])
def api_get_utts(cid):
    db   = get_db()
    rows = db.execute(
        "SELECT * FROM utterances WHERE conversation_id=? ORDER BY turn_index", (cid,)
    ).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/conversations/<int:cid>/utterances", methods=["POST"])
def api_add_utt(cid):
    data    = request.get_json(force=True)
    speaker = (data.get("speaker") or "Speaker A").strip()
    text    = (data.get("text") or "").strip()
    model   = data.get("model") or engine.active_model
    if not text:
        return jsonify({"error": "text required"}), 400

    pred = engine.predict(text, model)

    db  = get_db()
    max_turn = db.execute(
        "SELECT COALESCE(MAX(turn_index),-1) FROM utterances WHERE conversation_id=?", (cid,)
    ).fetchone()[0]
    turn = max_turn + 1
    cur  = db.execute(
        "INSERT INTO utterances (conversation_id,turn_index,speaker,text,emotion,confidence,model_used) VALUES (?,?,?,?,?,?,?)",
        (cid, turn, speaker, text, pred["emotion"], pred["confidence"], model)
    )
    uid = cur.lastrowid
    db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (cid,))
    db.commit()
    row = db.execute("SELECT * FROM utterances WHERE id=?", (uid,)).fetchone()
    db.close()
    return jsonify({**dict(row), **pred}), 201

@app.route("/api/utterances/<int:uid>", methods=["DELETE"])
def api_del_utt(uid):
    db = get_db()
    db.execute("DELETE FROM utterances WHERE id=?", (uid,))
    db.commit(); db.close()
    return jsonify({"deleted": uid})

@app.route("/api/utterances/<int:uid>", methods=["PUT"])
def api_edit_utt(uid):
    data    = request.get_json(force=True)
    text    = (data.get("text") or "").strip()
    speaker = (data.get("speaker") or "").strip()
    model   = data.get("model") or engine.active_model
    if not text:
        return jsonify({"error": "text required"}), 400

    pred = engine.predict(text, model)
    db   = get_db()
    db.execute(
        "UPDATE utterances SET text=?, speaker=?, emotion=?, confidence=?, model_used=? WHERE id=?",
        (text, speaker, pred["emotion"], pred["confidence"], model, uid)
    )
    db.commit()
    row = db.execute("SELECT * FROM utterances WHERE id=?", (uid,)).fetchone()
    db.close()
    return jsonify({**dict(row), **pred})

# ─────────────────────────────────────────────────────────────────────────────
# API — ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/conversations/<int:cid>/analyse", methods=["POST"])
def api_analyse(cid):
    data  = request.get_json(force=True) or {}
    model = data.get("model") or engine.active_model

    db   = get_db()
    rows = db.execute(
        "SELECT * FROM utterances WHERE conversation_id=? ORDER BY turn_index", (cid,)
    ).fetchall()
    if not rows:
        db.close()
        return jsonify({"error": "No utterances to analyse"}), 400

    utts = [{"utterance_id": r["id"], "turn_index": r["turn_index"],
              "speaker": r["speaker"], "text": r["text"]} for r in rows]

    result = engine.analyse_conversation(utts, model)

    for u in result["utterances"]:
        db.execute(
            "UPDATE utterances SET emotion=?, confidence=?, model_used=? WHERE id=?",
            (u["emotion"], u["confidence"], model, u["utterance_id"])
        )

    db.execute("DELETE FROM emotion_cause_pairs WHERE conversation_id=?", (cid,))
    for p in result["pairs"]:
        db.execute(
            """INSERT INTO emotion_cause_pairs
               (conversation_id,emotion_utterance_id,cause_utterance_id,emotion,cause_span,confidence,explanation)
               VALUES (?,?,?,?,?,?,?)""",
            (cid, p["emotion_utterance_id"], p["cause_utterance_id"],
             p["emotion"], p["cause_span"], p["confidence"], p["explanation"])
        )

    m = engine.metrics.get(model, {})
    db.execute(
        "INSERT INTO analysis_runs (conversation_id,model_used,pairs_found,accuracy,f1_score) VALUES (?,?,?,?,?)",
        (cid, model, len(result["pairs"]), m.get("accuracy", 0), m.get("f1", 0))
    )
    db.commit(); db.close()
    return jsonify(result)

@app.route("/api/conversations/<int:cid>/pairs", methods=["GET"])
def api_get_pairs(cid):
    db = get_db()
    pairs = db.execute(
        """SELECT ecp.*,
               eu.text AS emotion_text, eu.speaker AS emotion_speaker, eu.turn_index AS etn,
               cu.text AS cause_text,   cu.speaker AS cause_speaker,   cu.turn_index AS ctn
           FROM emotion_cause_pairs ecp
           JOIN utterances eu ON ecp.emotion_utterance_id=eu.id
           JOIN utterances cu ON ecp.cause_utterance_id=cu.id
           WHERE ecp.conversation_id=? ORDER BY eu.turn_index""",
        (cid,)
    ).fetchall()
    db.close()
    return jsonify([dict(p) for p in pairs])

# ─────────────────────────────────────────────────────────────────────────────
# API — QUICK ANALYSE (no DB save)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/quick-analyse", methods=["POST"])
def api_quick_analyse():
    data  = request.get_json(force=True)
    raw   = data.get("utterances", [])
    model = data.get("model") or engine.active_model
    if not raw:
        return jsonify({"error": "No utterances"}), 400
    utts = [{"turn_index": i, "speaker": u.get("speaker", f"S{i}"), "text": u["text"].strip()}
            for i, u in enumerate(raw) if u.get("text", "").strip()]
    return jsonify(engine.analyse_conversation(utts, model))

# ─────────────────────────────────────────────────────────────────────────────
# API — PREDICT SINGLE
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data  = request.get_json(force=True)
    text  = (data.get("text") or "").strip()
    model = data.get("model") or engine.active_model
    if not text:
        return jsonify({"error": "text required"}), 400
    all_preds = engine.predict_all(text)
    return jsonify(all_preds)

# ─────────────────────────────────────────────────────────────────────────────
# API — MODEL METRICS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(engine.get_all_metrics())

@app.route("/api/metrics/comparison", methods=["GET"])
def api_comparison():
    return jsonify(engine.get_comparison_table())

@app.route("/api/analytics/global", methods=["GET"])
def api_global_analytics():
    db = get_db()
    emo_dist = db.execute(
        "SELECT emotion, COUNT(*) AS cnt FROM utterances GROUP BY emotion"
    ).fetchall()
    daily = db.execute(
        """SELECT DATE(created_at) AS day, emotion, COUNT(*) AS cnt
           FROM utterances GROUP BY day, emotion ORDER BY day"""
    ).fetchall()
    top_conv = db.execute(
        """SELECT c.title, COUNT(ecp.id) AS pairs
           FROM emotion_cause_pairs ecp JOIN conversations c ON ecp.conversation_id=c.id
           GROUP BY ecp.conversation_id ORDER BY pairs DESC LIMIT 5"""
    ).fetchall()
    db.close()
    return jsonify({
        "emotion_distribution": [dict(r) for r in emo_dist],
        "daily_trend":          [dict(r) for r in daily],
        "top_conversations":    [dict(r) for r in top_conv],
        "comparison":           engine.get_comparison_table(),
        "model_metrics":        engine.get_all_metrics(),
    })

# ─────────────────────────────────────────────────────────────────────────────
# API — LOAD SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/load-samples", methods=["POST"])
def api_load_samples():
    sample_path = os.path.join(os.path.dirname(__file__), "sample_conversations.json")
    if not os.path.exists(sample_path):
        return jsonify({"error": "sample_conversations.json not found"}), 404
    with open(sample_path, encoding="utf-8") as f:
        samples = json.load(f)

    db      = get_db()
    created = []
    for conv in samples:
        cur = db.execute("INSERT INTO conversations (title) VALUES (?)", (conv["title"],))
        cid = cur.lastrowid
        for i, u in enumerate(conv["utterances"]):
            pred = engine.predict(u["text"])
            db.execute(
                "INSERT INTO utterances (conversation_id,turn_index,speaker,text,emotion,confidence,model_used) VALUES (?,?,?,?,?,?,?)",
                (cid, i, u["speaker"], u["text"], pred["emotion"], pred["confidence"], engine.active_model)
            )
        created.append({"id": cid, "title": conv["title"]})
    db.commit(); db.close()
    return jsonify({"loaded": len(created), "conversations": created})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
