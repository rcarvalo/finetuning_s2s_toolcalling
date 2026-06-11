#!/usr/bin/env python3
"""Démo web speech-to-speech : micro du navigateur → LFM2.5-Audio → réponse vocale.

Push-to-talk (maintenir le bouton, parler, relâcher) ; backend au choix dans
l'UI : liquid-audio (S2S natif) ou vLLM-Omni (transcription liquid + génération
vLLM tant que l'entrée audio du plugin n'est pas câblée).

    python scripts/s2s_web_demo.py --backend both --port 7860
    # puis ouvrir http://<hôte>:7860  (micro : nécessite localhost ou HTTPS)

Pas de WebRTC ici : l'audio est envoyé en HTTP une fois l'enregistrement fini
(latence = génération complète). Le streaming temps réel (WebRTC/WebSocket,
réponse pendant la génération) demande le chemin serving de vLLM-Omni, pas
l'API offline — cf. rapport.
"""

import argparse
import base64
import io
import threading
import time
import wave
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2s_demo import SR_OUT, LiquidBackend, VllmBackend  # noqa: E402

HTML = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>LFM2.5-Audio S2S</title>
<style>
 body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
 h1{font-size:1.2rem} .row{margin:.8rem 0}
 #talk{font-size:1.1rem;padding:1rem 2rem;border-radius:999px;border:0;background:#2563eb;color:#fff;cursor:pointer}
 #talk.rec{background:#dc2626}
 #log{border:1px solid #333;border-radius:8px;padding:1rem;min-height:8rem}
 .u{color:#93c5fd}.a{color:#86efac}.m{color:#999;font-size:.85rem}
 input[type=text]{width:70%;padding:.5rem;background:#222;color:#eee;border:1px solid #444;border-radius:6px}
 button{cursor:pointer}
</style></head><body>
<h1>🎙️ LFM2.5-Audio — speech to speech</h1>
<div class="row">Backend :
 <label><input type="radio" name="bk" value="liquid" __LIQ__> liquid-audio</label>
 <label><input type="radio" name="bk" value="vllm" __VLLM__> vLLM-Omni</label>
 <button onclick="fetch('/reset',{method:'POST'}).then(()=>log('— historique vidé —','m'))">reset</button>
</div>
<div class="row"><button id="talk">🎤 Maintenir pour parler</button>
 <span id="status" class="m"></span></div>
<div class="row"><input id="txt" type="text" placeholder="…ou tape un message">
 <button onclick="sendText()">Envoyer</button></div>
<div id="log"></div>
<script>
const logEl = document.getElementById('log'), st = document.getElementById('status');
function log(t, c){const d=document.createElement('div');d.className=c;d.textContent=t;logEl.appendChild(d);logEl.scrollTop=1e9}
function backend(){return document.querySelector('input[name=bk]:checked').value}

let ctx, src, proc, chunks;
async function startRec(){
  const stream = await navigator.mediaDevices.getUserMedia({audio:true});
  ctx = new AudioContext({sampleRate:16000});
  src = ctx.createMediaStreamSource(stream); chunks = [];
  proc = ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess = e => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  st.textContent = 'enregistrement…';
}
function wavBlob(){
  const n = chunks.reduce((s,c)=>s+c.length,0), pcm = new Int16Array(n);
  let o=0; for(const c of chunks) for(const v of c) pcm[o++] = Math.max(-1,Math.min(1,v))*32767;
  const buf = new ArrayBuffer(44+pcm.length*2), dv = new DataView(buf);
  const w=(p,s)=>{for(let i=0;i<s.length;i++)dv.setUint8(p+i,s.charCodeAt(i))};
  w(0,'RIFF'); dv.setUint32(4,36+pcm.length*2,true); w(8,'WAVEfmt '); dv.setUint32(16,16,true);
  dv.setUint16(20,1,true); dv.setUint16(22,1,true); dv.setUint32(24,16000,true);
  dv.setUint32(28,32000,true); dv.setUint16(32,2,true); dv.setUint16(34,16,true);
  w(36,'data'); dv.setUint32(40,pcm.length*2,true);
  new Int16Array(buf,44).set(pcm);
  return new Blob([buf],{type:'audio/wav'});
}
async function stopRec(){
  proc.disconnect(); src.disconnect(); await ctx.close();
  st.textContent = 'génération…';
  const fd = new FormData();
  fd.append('audio', wavBlob(), 'mic.wav'); fd.append('backend', backend());
  await send(fd);
}
async function sendText(){
  const t = document.getElementById('txt').value.trim(); if(!t) return;
  document.getElementById('txt').value=''; log('👤 '+t,'u');
  const fd = new FormData(); fd.append('text', t); fd.append('backend', backend());
  st.textContent='génération…'; await send(fd);
}
async function send(fd){
  try{
    const r = await fetch('/talk',{method:'POST',body:fd});
    const j = await r.json();
    if(j.error){log('⚠️ '+j.error,'m'); st.textContent=''; return}
    if(j.text_user) log('👤 '+j.text_user+'  (transcrit)','u');
    log('🤖 '+j.text,'a'); log(j.timing,'m');
    if(j.audio) new Audio('data:audio/wav;base64,'+j.audio).play();
  }catch(e){log('⚠️ '+e,'m')}
  st.textContent='';
}
const btn = document.getElementById('talk');
const down = e=>{e.preventDefault(); btn.classList.add('rec'); startRec()};
const up   = e=>{e.preventDefault(); if(!ctx||ctx.state==='closed')return; btn.classList.remove('rec'); stopRec()};
btn.addEventListener('mousedown',down); btn.addEventListener('mouseup',up);
btn.addEventListener('touchstart',down); btn.addEventListener('touchend',up);
</script></body></html>"""

TRANSCRIBE_SYSTEM = "Transcribe the user's speech verbatim. Respond with the transcription only, as text."


def wav_to_b64(wav: np.ndarray, rate: int = SR_OUT) -> str:
    buf = io.BytesIO()
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32_767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def build_app(backends: dict, lock: threading.Lock):
    from typing import Optional

    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI()
    html = (HTML
            .replace("__LIQ__", "checked" if "liquid" in backends else "disabled")
            .replace("__VLLM__", "checked" if "vllm" in backends and "liquid" not in backends
                     else ("" if "vllm" in backends else "disabled")))

    def transcribe(audio_path: Path) -> str:
        """ASR via le modèle liquid (tour isolé, hors historique)."""
        from liquid_audio import ChatState
        liq = backends["liquid"]
        chat = ChatState(liq.proc)
        chat.new_turn("system"); chat.add_text(TRANSCRIBE_SYSTEM); chat.end_turn()
        saved, liq.chat = liq.chat, chat
        try:
            txt, _wav, _m = liq.reply(audio_path=audio_path, max_new_tokens=128)
        finally:
            liq.chat = saved
        return txt

    @app.get("/")
    def index():
        return HTMLResponse(html)

    @app.post("/reset")
    def reset():
        with lock:
            for b in backends.values():
                b.reset()
        return {"ok": True}

    @app.post("/talk")
    def talk(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None),
             backend: str = Form(...)):
        if backend not in backends:
            return JSONResponse({"error": f"backend {backend!r} non chargé"})
        b = backends[backend]
        tmp: Path | None = None
        if audio is not None:
            tmp = Path(f"/tmp/s2s_in_{time.time_ns()}.wav")
            tmp.write_bytes(audio.file.read())

        text_user = None
        try:
            with lock:
                if backend == "vllm" and tmp is not None:
                    if "liquid" not in backends:
                        return JSONResponse({"error": "entrée micro avec vllm : lancez --backend both "
                                                      "(transcription via liquid)"})
                    text_user = transcribe(tmp)
                    txt, wav, m = b.reply(text=text_user)
                else:
                    txt, wav, m = b.reply(text=text, audio_path=tmp)
        except Exception as e:  # surface l'erreur dans l'UI plutôt qu'un 500 nu
            return JSONResponse({"error": f"{type(e).__name__}: {e}"})
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

        dur = wav.size / SR_OUT if wav is not None and wav.size else 0.0
        ttfa = f"TTFA {m['ttfa_s']:.2f}s · " if m.get("ttfa_s") else ""
        timing = f"[{b.name}] {ttfa}total {m['total_s']:.2f}s · {dur:.1f}s d'audio"
        return {"text": txt, "text_user": text_user, "timing": timing,
                "audio": wav_to_b64(wav) if dur else None}

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["liquid", "vllm", "both"], default="liquid")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    backends: dict = {}
    if args.backend in ("liquid", "both"):
        backends["liquid"] = LiquidBackend("/workspace/models/LFM2.5-Audio-1.5B")
    if args.backend in ("vllm", "both"):
        backends["vllm"] = VllmBackend("/workspace/models/lfm25_audio_omni")

    app = build_app(backends, threading.Lock())
    print(f"\n▶ démo prête : http://localhost:{args.port}  (micro : localhost ou HTTPS requis)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()