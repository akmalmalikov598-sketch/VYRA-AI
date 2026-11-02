
import os
import json
import uuid
import time
import hmac
import hashlib
import secrets
import sqlite3
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

# =========================================================
# VYRA AI | CONFIG
# =========================================================

APP_NAME = "VYRA AI"
BASE = Path(os.getenv("DATA_DIR", "data"))
VIDEO_DIR = BASE / "videos"
DB = BASE / "vyra.db"

BASE.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@vyra.local").lower()

jobs_lock = threading.Lock()
jobs = []
worker_running = False

# =========================================================
# DATABASE
# =========================================================

def connect():
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            prompt TEXT NOT NULL,
            style TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            video TEXT,
            error TEXT,
            created TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS project_user
        ON projects(user_id);
        """)


# =========================================================
# SECURITY
# =========================================================

def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    result = hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=2**14,
        r=8,
        p=1
    )
    return salt.hex() + ":" + result.hex()


def verify_password(password, stored):
    try:
        salt, original = stored.split(":")
        candidate = hash_password(password, bytes.fromhex(salt))
        return hmac.compare_digest(candidate, stored)
    except Exception:
        return False


def current_user(request):
    uid = request.session.get("uid")

    if not uid:
        return None

    with connect() as db:
        return db.execute(
            "SELECT id,name,email FROM users WHERE id=?",
            (uid,)
        ).fetchone()


def require_user(request):
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Avval tizimga kiring.")
    return user


def safe_filename(value):
    return Path(value).name


# =========================================================
# AI VIDEO ENGINE
# =========================================================

def generate_real_video(prompt, style, project_id):
    """
    Haqiqiy Wan 2.1 video generatsiyasi.
    Render Free CPU serverida CUDA GPU bo'lmasa ishlamaydi.
    """

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU topilmadi. Wan 2.1 uchun CUDA GPU worker kerak. "
            "Render Free serveri video modelini ishlata olmaydi."
        )

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=dtype
    )

    pipe.enable_model_cpu_offload()

    styles = {
        "Cinematic": "cinematic film, dramatic lighting, smooth camera movement",
        "Realistic": "photorealistic, realistic physics, natural lighting",
        "Animation": "beautiful 3D animation, animated movie",
        "Fantasy": "epic fantasy, magical atmosphere, cinematic lighting",
        "Sci-Fi": "futuristic science fiction, advanced technology"
    }

    final_prompt = (
        prompt + ", " + styles.get(style, styles["Cinematic"])
    )

    result = pipe(
        prompt=final_prompt,
        negative_prompt="blurry, distorted, low quality, flickering",
        height=480,
        width=832,
        num_frames=17,
        num_inference_steps=20,
        guidance_scale=5.0
    )

    output = VIDEO_DIR / f"{project_id}.mp4"

    export_to_video(
        result.frames[0],
        str(output),
        fps=16
    )

    del pipe, vae
    torch.cuda.empty_cache()

    return output


# =========================================================
# JOB QUEUE
# =========================================================

def update_project(project_id, status, video=None, error=None):
    with connect() as db:
        db.execute(
            """
            UPDATE projects
            SET status=?, video=?, error=?
            WHERE id=?
            """,
            (status, video, error, project_id)
        )


def job_worker():
    global worker_running

    while True:
        with jobs_lock:
            if not jobs:
                worker_running = False
                return
            job = jobs.pop(0)

        project_id = job["id"]

        try:
            update_project(project_id, "processing")

            video = generate_real_video(
                job["prompt"],
                job["style"],
                project_id
            )

            update_project(
                project_id,
                "completed",
                video.name
            )

        except Exception as error:
            update_project(
                project_id,
                "failed",
                error=str(error)[:1000]
            )


def enqueue(job):
    global worker_running

    with jobs_lock:
        jobs.append(job)

        if not worker_running:
            worker_running = True
            threading.Thread(
                target=job_worker,
                daemon=True
            ).start()


# =========================================================
# FASTAPI
# =========================================================

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="VYRA AI",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET,
    session_cookie="vyra_session",
    same_site="lax",
    https_only=True,
    max_age=60 * 60 * 24 * 7
)


# =========================================================
# HTML INTERFACE
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#090912">
<title>VYRA AI — Create Beyond Limits</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
--bg:#090912;--panel:#11111e;--panel2:#171729;
--line:#292943;--text:#f7f5ff;--muted:#9291ac;
--purple:#9064ff;--blue:#536cff;--pink:#db70ff;
}
*{box-sizing:border-box}
body{
margin:0;background:var(--bg);color:var(--text);
font-family:Inter,Arial,sans-serif;
}
button,input,textarea,select{font:inherit}
button{cursor:pointer}
.app{min-height:100vh}
.nav{
height:75px;border-bottom:1px solid var(--line);
display:flex;align-items:center;justify-content:space-between;
padding:0 5%;position:sticky;top:0;background:#090912ed;
backdrop-filter:blur(20px);z-index:5;
}
.logo{
font-size:25px;font-weight:900;letter-spacing:-1.5px;
background:linear-gradient(90deg,#ae91ff,#63b5ff,#e77cff);
background-clip:text;-webkit-background-clip:text;color:transparent;
}
.navright{display:flex;align-items:center;gap:15px}
.avatar{
width:36px;height:36px;border-radius:50%;
display:grid;place-items:center;
background:linear-gradient(135deg,var(--purple),var(--blue));
font-weight:800;
}
button{
border:0;border-radius:11px;padding:12px 17px;
background:var(--panel2);color:white;
}
button:hover{filter:brightness(1.2)}
.primary{
background:linear-gradient(110deg,#7950ed,#526aff);
font-weight:800;box-shadow:0 7px 25px #7450ed30;
}
.layout{
display:grid;grid-template-columns:230px minmax(0,1fr);
max-width:1600px;margin:auto;min-height:calc(100vh - 75px);
}
.sidebar{
border-right:1px solid var(--line);padding:28px 15px;
}
.side-label{
color:#777690;font-size:10px;font-weight:800;
letter-spacing:2px;padding:12px;
}
.side-item{
width:100%;display:flex;align-items:center;gap:12px;
padding:14px;border-radius:11px;color:#aaa9c4;
text-align:left;margin-bottom:5px;background:transparent;
}
.side-item.active,.side-item:hover{
background:#211b3c;color:white;
}
.main{padding:35px;min-width:0}
.hero{
border:1px solid #39305c;border-radius:23px;padding:35px;
background:
radial-gradient(ellipse at 90% 0%,#5932a64c,transparent 50%),
linear-gradient(120deg,#171329,#10101c);
margin-bottom:27px;position:relative;overflow:hidden;
}
.hero h1{font-size:clamp(30px,4vw,48px);margin:0 0 12px;letter-spacing:-2px}
.gradient{
background:linear-gradient(90deg,#b69aff,#71a7ff,#e78aff);
background-clip:text;-webkit-background-clip:text;color:transparent;
}
.hero p{color:#aaa8c4;max-width:550px;line-height:1.8}
.eyebrow{color:#b99bff;font-size:11px;font-weight:800;letter-spacing:2px}
.section-title{font-size:21px;font-weight:800;margin:27px 0 18px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.card{
background:var(--panel);border:1px solid var(--line);
border-radius:17px;padding:21px;min-width:0;
}
.card h3{margin:0 0 8px;font-size:16px}
.muted{color:var(--muted);font-size:13px;line-height:1.7}
label{display:block;color:#c6c3da;font-size:12px;font-weight:700;margin:17px 0 9px}
input,textarea,select{
width:100%;padding:14px;background:#0b0b16;
border:1px solid #30304b;color:white;border-radius:10px;
outline:none;
}
input:focus,textarea:focus,select:focus{border-color:#9470ff}
textarea{min-height:135px;resize:vertical;line-height:1.7}
.full{width:100%;margin-top:17px}
.preview{
min-height:290px;background:
radial-gradient(ellipse,#262044,#0b0b14 70%);
border:1px solid var(--line);border-radius:15px;
display:flex;align-items:center;justify-content:center;
overflow:hidden;
}
.preview video{width:100%;max-height:450px}
.empty{text-align:center;color:#87859e;padding:30px}
.empty .big{font-size:48px;margin-bottom:12px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:22px 0}
.stat{background:#141421;border:1px solid var(--line);border-radius:14px;padding:17px}
.stat strong{display:block;font-size:23px;margin-bottom:7px}
.stat span{font-size:11px;color:var(--muted)}
.project{
padding:16px;border:1px solid var(--line);border-radius:13px;
margin:10px 0;background:#11111d;
display:flex;justify-content:space-between;gap:12px;align-items:center;
}
.status{font-size:11px;color:#b9a4ff}
.hidden{display:none!important}
.auth{
min-height:100vh;display:grid;place-items:center;padding:20px;
background:radial-gradient(ellipse at 50% 0%,#312052,#090912 65%);
}
.authbox{
width:min(430px,100%);background:#11111f;
border:1px solid #34304d;border-radius:22px;padding:32px;
}
.authbox h1{text-align:center;font-size:32px}
.tabs{display:flex;gap:8px;margin:20px 0}
.tabs button{flex:1}
.notice{
padding:13px;border:1px solid #41365d;
background:#1c1830;border-radius:10px;
color:#c7b9f5;font-size:12px;line-height:1.7;margin-top:15px;
}
.mobile-menu{display:none}
footer{text-align:center;color:#62617c;font-size:11px;padding:35px}
@media(max-width:850px){
.layout{grid-template-columns:1fr}
.sidebar{display:flex;overflow:auto;border-right:0;border-bottom:1px solid var(--line);padding:8px}
.side-label{display:none}
.side-item{white-space:nowrap;width:auto;margin:0}
.main{padding:19px}
.nav{padding:0 18px}
.hero{padding:24px}
}
@media(max-width:540px){
.grid{grid-template-columns:1fr}
.stats{grid-template-columns:1fr}
.main{padding:13px}
.hero h1{font-size:31px}
.nav{height:65px}
}
</style>
</head>
<body>
<div id="auth" class="auth hidden">
<div class="authbox">
<div class="logo" style="text-align:center">VYRA AI</div>
<h1 id="authTitle">Create your world.</h1>
<p class="muted" style="text-align:center">AI VIDEO CREATION STUDIO</p>
<div class="tabs">
<button onclick="authMode('login')" id="loginTab" class="primary">Kirish</button>
<button onclick="authMode('register')" id="registerTab">Ro‘yxatdan o‘tish</button>
</div>
<div id="nameField" class="hidden">
<label>Ismingiz</label><input id="name" maxlength="60" placeholder="Ismingiz">
</div>
<label>Email</label><input id="email" type="email" maxlength="150" placeholder="you@example.com">
<label>Parol</label><input id="password" type="password" maxlength="128" placeholder="Kamida 8 ta belgi">
<button class="primary full" onclick="authenticate()" id="authButton">Davom etish</button>
<div id="authError" class="notice hidden"></div>
</div>
</div>

<div id="app" class="app hidden">
<header class="nav">
<div class="logo">VYRA AI</div>
<div class="navright">
<span id="welcome" class="muted"></span>
<div class="avatar" id="avatar">V</div>
<button onclick="logout()">Chiqish</button>
</div>
</header>

<div class="layout">
<aside class="sidebar">
<div class="side-label">WORKSPACE</div>
<button class="side-item active" onclick="showPage('studio',this)">✦ AI Studio</button>
<button class="side-item" onclick="showPage('projects',this)">▦ My Projects</button>
<button class="side-item" onclick="showPage('account',this)">◉ Account</button>
<div class="side-label">INFORMATION</div>
<button class="side-item" onclick="showPage('about',this)">ⓘ About VYRA</button>
</aside>

<main class="main">
<section id="studio">
<div class="hero">
<div class="eyebrow">THE FUTURE OF STORYTELLING</div>
<h1>Create beyond <span class="gradient">limits.</span></h1>
<p>G‘oyangizni yozing. Sahnalarni yarating. Hikoyangizni kinoga aylantiring.</p>
</div>

<div class="stats">
<div class="stat"><strong id="projectCount">0</strong><span>LOYIHALAR</span></div>
<div class="stat"><strong>AI</strong><span>VIDEO ENGINE</span></div>
<div class="stat"><strong>∞</strong><span>CREATIVE IDEAS</span></div>
</div>

<div class="section-title">✦ Create a new project</div>
<div class="grid">
<div class="card">
<h3>Video generator</h3>
<p class="muted">Har bir qatorga alohida sahna promptini yozing.</p>
<label>PROJECT NAME</label>
<input id="title" maxlength="100" value="My cinematic story">
<label>SCENE PROMPTS</label>
<textarea id="prompt" maxlength="5000" placeholder="A cinematic view of a futuristic city at sunset...

A flying vehicle moves between the skyscrapers...

The camera moves toward a beautiful glowing skyline..."></textarea>
<label>VISUAL STYLE</label>
<select id="style">
<option>Cinematic</option>
<option>Realistic</option>
<option>Animation</option>
<option>Fantasy</option>
<option>Sci-Fi</option>
</select>
<button class="primary full" onclick="createProject()">✦ GENERATE VIDEO</button>
<div class="notice">GPU ulanmagan bo‘lsa, vazifa navbatga qo‘yiladi va GPU mavjud emasligi haqida xabar ko‘rsatiladi. Pullik AI API ishlatilmaydi.</div>
</div>

<div class="card">
<h3>Video preview</h3>
<p class="muted">Yaratilgan videongiz shu yerda ko‘rinadi.</p>
<div class="preview" id="preview">
<div class="empty"><div class="big">▶</div>YOUR NEXT CREATION<br><small>VYRA AI STUDIO</small></div>
</div>
<div id="jobMessage" class="muted" style="margin-top:14px"></div>
</div>
</div>
</section>

<section id="projects" class="hidden">
<div class="hero"><div class="eyebrow">YOUR CREATIVE SPACE</div><h1>My <span class="gradient">Projects.</span></h1><p>Barcha loyihalaringiz bitta joyda.</p></div>
<div id="projectList"></div>
</section>

<section id="account" class="hidden">
<div class="hero"><div class="eyebrow">PERSONAL WORKSPACE</div><h1>Your <span class="gradient">Account.</span></h1></div>
<div class="card"><h3>Profil ma'lumotlari</h3><p id="profileName"></p><p id="profileEmail" class="muted"></p></div>
</section>

<section id="about" class="hidden">
<div class="hero"><div class="eyebrow">INDEPENDENT AI STUDIO</div><h1>About <span class="gradient">VYRA.</span></h1></div>
<div class="card">
<h3>Open-source AI platform</h3>
<p class="muted">VYRA AI mustaqil video yaratish platformasi. Maqsadimiz — ochiq modellar orqali matn va rasmlardan haqiqiy harakatlanuvchi videolar yaratish.</p>
<p class="muted">No paid AI API keys. GPU model execution is separate from the web application.</p>
</div>
</section>

<footer>VYRA AI · CREATE WORLDS. FRAME STORIES.</footer>
</main>
</div>
</div>

<script>
let mode='login';
let poller=null;

function authMode(value){
 mode=value;
 document.getElementById('nameField').classList.toggle('hidden',value!=='register');
 document.getElementById('authTitle').textContent=value==='login'?'Welcome back.':'Create your world.';
 document.getElementById('authButton').textContent=value==='login'?'Kirish':'Hisob yaratish';
 document.getElementById('loginTab').classList.toggle('primary',value==='login');
 document.getElementById('registerTab').classList.toggle('primary',value==='register');
 document.getElementById('authError').classList.add('hidden');
}

async function api(url,options={}){
 const response=await fetch(url,{
  credentials:'same-origin',
  ...options,
  headers:{'Content-Type':'application/json',...(options.headers||{})}
 });
 const data=await response.json().catch(()=>({detail:'Server xatosi'}));
 if(!response.ok)throw new Error(data.detail||'Xatolik yuz berdi');
 return data;
}

async function authenticate(){
 const error=document.getElementById('authError');
 error.classList.add('hidden');
 try{
  const data={
   name:document.getElementById('name').value,
   email:document.getElementById('email').value,
   password:document.getElementById('password').value
  };
  await api(mode==='login'?'/api/login':'/api/register',{
   method:'POST',body:JSON.stringify(data)
  });
  await boot();
 }catch(e){
  error.textContent=e.message;
  error.classList.remove('hidden');
 }
}

async function boot(){
 try{
  const user=await api('/api/me');
  document.getElementById('auth').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  document.getElementById('welcome').textContent=user.name;
  document.getElementById('avatar').textContent=user.name[0].toUpperCase();
  document.getElementById('profileName').textContent=user.name;
  document.getElementById('profileEmail').textContent=user.email;
  await loadProjects();
 }catch(e){
  document.getElementById('app').classList.add('hidden');
  document.getElementById('auth').classList.remove('hidden');
 }
}

async function logout(){
 await api('/api/logout',{method:'POST'});
 location.reload();
}

function showPage(page,button){
 ['studio','projects','account','about'].forEach(id=>{
  document.getElementById(id).classList.toggle('hidden',id!==page);
 });
 document.querySelectorAll('.side-item').forEach(x=>x.classList.remove('active'));
 if(button)button.classList.add('active');
 if(page==='projects')loadProjects();
}

async function createProject(){
 const message=document.getElementById('jobMessage');
 const prompt=document.getElementById('prompt').value;
 const title=document.getElementById('title').value;
 if(!prompt.trim()){message.textContent='Avval prompt yozing.';return}
 message.textContent='Loyiha yaratilmoqda...';
 try{
  const job=await api('/api/projects',{
   method:'POST',
   body:JSON.stringify({
    title,prompt,style:document.getElementById('style').value
   })
  });
  message.textContent='Vazifa: '+job.status+' | ID: '+job.id;
  watchJob(job.id);
  await loadProjects();
 }catch(e){message.textContent=e.message}
}

async function watchJob(id){
 if(poller)clearInterval(poller);
 poller=setInterval(async()=>{
  try{
   const job=await api('/api/projects/'+id);
   document.getElementById('jobMessage').textContent=
    'Holat: '+job.status+(job.error?' — '+job.error:'');
   if(job.status==='completed'){
    clearInterval(poller);
    const preview=document.getElementById('preview');
    preview.innerHTML='';
    const video=document.createElement('video');
    video.controls=true;
    video.src=job.video;
    preview.appendChild(video);
    const link=document.createElement('a');
    link.href=job.video;
    link.download='vyra-video.mp4';
    link.textContent='Videoni yuklab olish';
    link.style.display='block';
    link.style.marginTop='12px';
    document.getElementById('preview').after(link);
    await loadProjects();
   }
   if(job.status==='failed'){
    clearInterval(poller);
    await loadProjects();
   }
  }catch(e){clearInterval(poller)}
 },4000);
}

async function loadProjects(){
 try{
  const data=await api('/api/projects');
  document.getElementById('projectCount').textContent=data.length;
  const list=document.getElementById('projectList');
  list.innerHTML='';
  if(!data.length){
   list.textContent='Hali loyihalar mavjud emas.';
   return;
  }
  data.forEach(item=>{
   const div=document.createElement('div');
   div.className='project';
   const info=document.createElement('div');
   const name=document.createElement('strong');
   name.textContent=item.title;
   const sub=document.createElement('div');
   sub.className='muted';
   sub.textContent=item.created+' · '+item.status;
   info.append(name,sub);
   div.append(info);
   if(item.video){
    const link=document.createElement('a');
    link.href=item.video;
    link.textContent='Videoni ochish';
    link.style.color='#bba5ff';
    div.append(link);
   }
   list.append(div);
  });
 }catch(e){}
}

boot();
</script>
</body>
</html>
"""


# =========================================================
# WEB ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    return {"status": "online", "app": APP_NAME}


@app.post("/api/register")
async def register(request: Request):
    data = await request.json()

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not 2 <= len(name) <= 60:
        raise HTTPException(400, "Ism 2-60 belgi bo‘lishi kerak.")

    if "@" not in email or len(email) > 150:
        raise HTTPException(400, "Email noto‘g‘ri.")

    if len(password) < 8 or len(password) > 128:
        raise HTTPException(400, "Parol 8-128 belgi bo‘lishi kerak.")

    with connect() as db:
        try:
            cursor = db.execute(
                """
                INSERT INTO users(name,email,password,created)
                VALUES(?,?,?,?)
                """,
                (name, email, hash_password(password),
                 datetime.utcnow().isoformat())
            )
            uid = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Bu email oldin ro‘yxatdan o‘tgan.")

    request.session.clear()
    request.session["uid"] = uid

    return {"success": True}


@app.post("/api/login")
async def login(request: Request):
    data = await request.json()

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    with connect() as db:
        user = db.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(401, "Email yoki parol noto‘g‘ri.")

    request.session.clear()
    request.session["uid"] = user["id"]

    return {"success": True}


@app.post("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/me")
async def me(request: Request):
    user = require_user(request)
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"]
    }


@app.post("/api/projects")
async def create_project(request: Request):
    user = require_user(request)
    data = await request.json()

    title = str(data.get("title", "My project")).strip()[:100]
    prompt = str(data.get("prompt", "")).strip()
    style = str(data.get("style", "Cinematic"))

    allowed = ["Cinematic", "Realistic", "Animation", "Fantasy", "Sci-Fi"]

    if not prompt or len(prompt) > 5000:
        raise HTTPException(400, "Prompt 1-5000 belgi bo‘lishi kerak.")

    if style not in allowed:
        raise HTTPException(400, "Uslub noto‘g‘ri.")

    project_id = uuid.uuid4().hex

    with connect() as db:
        db.execute(
            """
            INSERT INTO projects
            (id,user_id,title,prompt,style,status,created)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                project_id,
                user["id"],
                title or "My project",
                prompt,
                style,
                "queued",
                datetime.utcnow().isoformat()
            )
        )

    enqueue({
        "id": project_id,
        "prompt": prompt,
        "style": style
    })

    return {"id": project_id, "status": "queued"}


@app.get("/api/projects")
async def list_projects(request: Request):
    user = require_user(request)

    with connect() as db:
        rows = db.execute(
            """
            SELECT id,title,status,video,created
            FROM projects
            WHERE user_id=?
            ORDER BY created DESC
            LIMIT 100
            """,
            (user["id"],)
        ).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "status": row["status"],
            "created": row["created"][:16],
            "video": (
                f"/api/videos/{row['id']}"
                if row["video"] else None
            )
        }
        for row in rows
    ]


@app.get("/api/projects/{project_id}")
async def project_status(project_id: str, request: Request):
    user = require_user(request)

    with connect() as db:
        row = db.execute(
            """
            SELECT *
            FROM projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user["id"])
        ).fetchone()

    if not row:
        raise HTTPException(404, "Loyiha topilmadi.")

    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "error": row["error"],
        "video": (
            f"/api/videos/{row['id']}"
            if row["video"] else None
        )
    }


@app.get("/api/videos/{project_id}")
async def download_video(project_id: str, request: Request):
    user = require_user(request)

    with connect() as db:
        row = db.execute(
            """
            SELECT video,status
            FROM projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user["id"])
        ).fetchone()

    if not row or row["status"] != "completed" or not row["video"]:
        raise HTTPException(404, "Video mavjud emas.")

    filename = safe_filename(row["video"])
    path = VIDEO_DIR / filename

    if not path.is_file():
        raise HTTPException(404, "Video fayli topilmadi.")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
    )
