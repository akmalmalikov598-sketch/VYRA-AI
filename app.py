
import os
import re
import json
import uuid
import time
import hmac
import hashlib
import secrets
import sqlite3
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware

# =========================================================
# VYRA AI — OPEN SOURCE VIDEO PLATFORM
# Render backend + Google Colab GPU worker
# No paid AI API keys
# =========================================================

APP_NAME = "VYRA AI"

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
VIDEO_DIR = DATA_DIR / "videos"
DB_PATH = DATA_DIR / "vyra.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

SESSION_SECRET = os.getenv("SESSION_SECRET", "")
WORKER_SECRET = os.getenv("WORKER_SECRET", "")

if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET environment variable kerak.")

if not WORKER_SECRET or len(WORKER_SECRET) < 32:
    raise RuntimeError("WORKER_SECRET kamida 32 belgidan iborat bo‘lishi kerak.")

MAX_VIDEO_SIZE = 100 * 1024 * 1024
MAX_SCENES = 3
MAX_PROMPT_LENGTH = 5000

STYLES = [
    "Cinematic",
    "Realistic",
    "Animation",
    "Fantasy",
    "Sci-Fi"
]

# =========================================================
# DATABASE
# =========================================================

@contextmanager
def database():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=30000")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    with database() as db:
        db.execute("PRAGMA journal_mode=WAL")

        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            prompts TEXT NOT NULL,
            style TEXT NOT NULL,
            duration INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            video TEXT,
            error TEXT,
            created REAL NOT NULL,
            updated REAL NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS projects_user_idx
        ON projects(user_id, created);

        CREATE INDEX IF NOT EXISTS projects_status_idx
        ON projects(status, created);
        """)


# =========================================================
# PASSWORD SECURITY
# =========================================================

def password_hash(password):
    salt = secrets.token_bytes(16)
    result = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1
    )
    return salt.hex() + ":" + result.hex()


def password_verify(password, stored):
    try:
        salt, expected = stored.split(":")
        result = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt),
            n=2**14,
            r=8,
            p=1
        )
        return hmac.compare_digest(result.hex(), expected)
    except Exception:
        return False


def get_user(request):
    user_id = request.session.get("user_id")

    if not user_id:
        return None

    with database() as db:
        return db.execute(
            "SELECT id,name,email FROM users WHERE id=?",
            (user_id,)
        ).fetchone()


def auth_required(request):
    user = get_user(request)
    if not user:
        raise HTTPException(401, "Tizimga kiring.")
    return user


def check_worker(request):
    supplied = request.headers.get("X-Worker-Secret", "")

    if not hmac.compare_digest(supplied, WORKER_SECRET):
        raise HTTPException(403, "Worker authentication failed.")


# =========================================================
# APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="VYRA AI",
    description="Open-source AI video creation platform",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="vyra_session",
    same_site="lax",
    https_only=True,
    max_age=604800
)


# =========================================================
# PREMIUM WEB DESIGN
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#080811">
<title>VYRA AI — Video Creation Studio</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
--bg:#080811;
--surface:#10101c;
--surface2:#171727;
--border:#292940;
--text:#f8f7ff;
--muted:#9392ac;
--purple:#966aff;
--blue:#5687ff;
--pink:#e879ff;
--green:#48d6a0;
}
*{box-sizing:border-box}
body{
margin:0;background:var(--bg);color:var(--text);
font-family:Inter,Arial,sans-serif;
}
button,input,textarea,select{font:inherit}
button{cursor:pointer;color:white}
.hidden{display:none!important}
a{color:#bba5ff;text-decoration:none}
.nav{
height:72px;position:sticky;top:0;z-index:10;
display:flex;align-items:center;justify-content:space-between;
padding:0 5%;background:#080811ed;
backdrop-filter:blur(20px);
border-bottom:1px solid var(--border);
}
.logo{
font-size:27px;font-weight:900;letter-spacing:-1.7px;
background:linear-gradient(100deg,#c1a3ff,#78a8ff,#f08aff);
background-clip:text;-webkit-background-clip:text;color:transparent;
}
.nav-right{display:flex;align-items:center;gap:15px}
.pill{
border:1px solid #393151;border-radius:50px;
padding:7px 12px;color:#c8b7ff;font-size:11px;
}
.layout{
max-width:1600px;margin:auto;
display:grid;grid-template-columns:230px minmax(0,1fr);
min-height:calc(100vh - 72px);
}
.sidebar{border-right:1px solid var(--border);padding:24px 13px}
.side-title{
font-size:10px;letter-spacing:2px;color:#77768d;
font-weight:800;padding:12px;
}
.side-button{
display:block;width:100%;text-align:left;
padding:13px;border-radius:11px;
background:transparent;color:#a5a3bd;margin-bottom:5px;
border:0;
}
.side-button:hover,.side-button.active{
background:#211b3b;color:white;
}
main{padding:32px;min-width:0}
.hero{
padding:37px;border:1px solid #3c315d;
border-radius:23px;margin-bottom:24px;
background:
radial-gradient(ellipse at 90% 0%,#7143b744,transparent 50%),
linear-gradient(125deg,#1a1530,#11111e);
}
.eyebrow{
color:#b69bff;font-size:10px;
font-weight:900;letter-spacing:2.5px;
}
.hero h1{
font-size:clamp(33px,5vw,57px);
letter-spacing:-2.8px;margin:13px 0;
}
.gradient{
background:linear-gradient(90deg,#c1a3ff,#73a9ff,#e98aff);
background-clip:text;-webkit-background-clip:text;color:transparent;
}
.hero p,.muted{color:var(--muted);line-height:1.8;font-size:13px}
.section-title{font-size:21px;font-weight:800;margin:25px 0 17px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{
background:var(--surface);border:1px solid var(--border);
border-radius:18px;padding:21px;min-width:0;
}
.card h3{margin:0 0 8px}
label{
display:block;font-size:11px;font-weight:800;
color:#c5c1db;letter-spacing:.5px;margin:18px 0 9px;
}
input,textarea,select{
width:100%;background:#0a0a15;
border:1px solid #30304a;border-radius:10px;
padding:14px;color:white;outline:none;
}
input:focus,textarea:focus,select:focus{border-color:var(--purple)}
textarea{min-height:145px;resize:vertical;line-height:1.7}
.primary{
background:linear-gradient(110deg,#8255f1,#477bfa);
border:0;border-radius:11px;padding:15px;
font-weight:900;
box-shadow:0 8px 25px #704cdb33;
}
.primary:hover{filter:brightness(1.15)}
.full{width:100%;margin-top:17px}
.preview{
min-height:300px;border:1px solid var(--border);
border-radius:15px;background:
radial-gradient(ellipse,#282044,#0a0a14 72%);
display:grid;place-items:center;overflow:hidden;
}
.preview video{width:100%;max-height:460px}
.placeholder{text-align:center;color:#8b89a4;padding:25px}
.placeholder b{display:block;font-size:45px;margin-bottom:15px}
.notice{
background:#1b1730;border:1px solid #43365e;
padding:13px;border-radius:11px;color:#c6b9ef;
font-size:11px;line-height:1.7;margin-top:15px;
}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}
.stat{
padding:16px;border-radius:14px;
background:#12121f;border:1px solid var(--border);
}
.stat b{display:block;font-size:24px;margin-bottom:7px}
.stat span{font-size:10px;color:var(--muted)}
.project{
display:flex;align-items:center;justify-content:space-between;
gap:15px;padding:17px;border:1px solid var(--border);
background:var(--surface);border-radius:13px;margin:10px 0;
}
.status{font-size:11px;color:#c4a8ff;margin-top:5px}
.auth{
min-height:100vh;display:grid;place-items:center;
padding:20px;background:
radial-gradient(ellipse at top,#372052,#080811 70%);
}
.auth-card{
width:min(430px,100%);padding:32px;
background:#11111f;border:1px solid #39304e;border-radius:22px;
}
.auth-card h1{text-align:center}
.auth-tabs{display:flex;gap:8px;margin:20px 0}
.auth-tabs button{flex:1}
.auth-error{color:#ff9baf;margin-top:12px;font-size:12px}
footer{text-align:center;color:#67657f;padding:35px;font-size:10px}
@media(max-width:850px){
.layout{grid-template-columns:1fr}
.sidebar{display:flex;overflow-x:auto;padding:8px;border-right:0;border-bottom:1px solid var(--border)}
.side-title{display:none}
.side-button{white-space:nowrap;width:auto}
main{padding:18px}
}
@media(max-width:560px){
.grid{grid-template-columns:1fr}
.stats{grid-template-columns:1fr}
main{padding:12px}
.hero{padding:24px}
.nav{padding:0 15px;height:64px}
}
</style>
</head>

<body>

<div id="auth" class="auth hidden">
<div class="auth-card">
<div class="logo" style="text-align:center">VYRA AI</div>
<h1 id="authTitle">Welcome back.</h1>
<p class="muted" style="text-align:center">YOUR AI VIDEO STUDIO</p>

<div class="auth-tabs">
<button id="loginTab" class="primary" onclick="setMode('login')">Kirish</button>
<button id="registerTab" onclick="setMode('register')">Ro‘yxatdan o‘tish</button>
</div>

<div id="nameBox" class="hidden">
<label>ISMINGIZ</label>
<input id="name" maxlength="60" placeholder="Ismingiz">
</div>

<label>EMAIL</label>
<input id="email" type="email" maxlength="150" placeholder="email@example.com">

<label>PAROL</label>
<input id="password" type="password" maxlength="128" placeholder="Kamida 8 belgi">

<button class="primary full" id="authButton" onclick="authenticate()">Kirish</button>
<div id="authError" class="auth-error"></div>
</div>
</div>

<div id="application" class="hidden">

<header class="nav">
<div class="logo">VYRA AI</div>
<div class="nav-right">
<span class="pill">OPEN SOURCE STUDIO</span>
<span id="userName" class="muted"></span>
<button onclick="logout()">Chiqish</button>
</div>
</header>

<div class="layout">

<aside class="sidebar">
<div class="side-title">WORKSPACE</div>
<button class="side-button active" onclick="page('studio',this)">✦ AI Studio</button>
<button class="side-button" onclick="page('projects',this)">▦ My Projects</button>
<button class="side-button" onclick="page('account',this)">◉ Account</button>
<div class="side-title">INFORMATION</div>
<button class="side-button" onclick="page('about',this)">ⓘ About</button>
</aside>

<main>

<section id="studio">

<div class="hero">
<div class="eyebrow">THE FUTURE OF STORYTELLING</div>
<h1>Create beyond <span class="gradient">limits.</span></h1>
<p>G‘oyangizni yozing. Har bir sahnani yarating. Hikoyangizni videoga aylantiring.</p>
</div>

<div class="stats">
<div class="stat"><b id="count">0</b><span>LOYIHALAR</span></div>
<div class="stat"><b>WAN 2.1</b><span>VIDEO ENGINE</span></div>
<div class="stat"><b>GPU</b><span>COLAB WORKER</span></div>
</div>

<div class="section-title">✦ Create a new project</div>

<div class="grid">

<div class="card">
<h3>AI Video Generator</h3>
<p class="muted">Har bir qatorda alohida sahna promptini yozing.</p>

<label>PROJECT NAME</label>
<input id="title" maxlength="100" value="My cinematic story">

<label>SCENE PROMPTS</label>
<textarea id="prompts" maxlength="5000"
placeholder="A cinematic view of a futuristic city at sunset...

A flying vehicle moves between beautiful skyscrapers...

The camera moves toward a glowing skyline..."></textarea>

<label>VISUAL STYLE</label>
<select id="style">
<option>Cinematic</option>
<option>Realistic</option>
<option>Animation</option>
<option>Fantasy</option>
<option>Sci-Fi</option>
</select>

<label>HAR BIR SAHNA DAVOMIYLIGI</label>
<select id="duration">
<option value="2">2 soniya</option>
<option value="3">3 soniya</option>
<option value="4">4 soniya</option>
<option value="5">5 soniya</option>
</select>

<button class="primary full" onclick="createProject()">✦ GENERATE VIDEO</button>

<div class="notice">
Colab GPU ishga tushgan bo‘lishi kerak. GPU worker vazifani olgach, haqiqiy video yaratadi.
</div>
</div>

<div class="card">
<h3>Video Preview</h3>
<p class="muted">Tayyor videongiz shu yerda ko‘rinadi.</p>
<div id="preview" class="preview">
<div class="placeholder"><b>▶</b>YOUR NEXT CREATION<br>VYRA AI STUDIO</div>
</div>
<div id="jobStatus" class="muted" style="margin-top:15px">Tizim tayyor.</div>
</div>

</div>
</section>

<section id="projects" class="hidden">
<div class="hero">
<div class="eyebrow">YOUR CREATIVE SPACE</div>
<h1>My <span class="gradient">Projects.</span></h1>
<p>Barcha yaratilgan loyihalaringiz.</p>
</div>
<div id="projectList"></div>
</section>

<section id="account" class="hidden">
<div class="hero">
<div class="eyebrow">PERSONAL WORKSPACE</div>
<h1>Your <span class="gradient">Account.</span></h1>
</div>
<div class="card">
<h3>Profil</h3>
<p id="profileName"></p>
<p id="profileEmail" class="muted"></p>
</div>
</section>

<section id="about" class="hidden">
<div class="hero">
<div class="eyebrow">INDEPENDENT AI STUDIO</div>
<h1>About <span class="gradient">VYRA.</span></h1>
</div>
<div class="card">
<h3>Open-source AI video creation</h3>
<p class="muted">VYRA AI ochiq video modellarini o‘z GPU worker'ida ishga tushiradi.</p>
<p class="muted">No paid AI APIs. Render backend + Google Colab GPU.</p>
</div>
</section>

<footer>VYRA AI · CREATE WORLDS. FRAME STORIES.</footer>
</main>
</div>
</div>

<script>
let authMode='login';
let pollTimer=null;

function setMode(mode){
 authMode=mode;
 document.getElementById('nameBox').classList.toggle('hidden',mode!=='register');
 document.getElementById('authTitle').textContent=mode==='login'?'Welcome back.':'Create your world.';
 document.getElementById('authButton').textContent=mode==='login'?'Kirish':'Hisob yaratish';
 document.getElementById('loginTab').classList.toggle('primary',mode==='login');
 document.getElementById('registerTab').classList.toggle('primary',mode==='register');
 document.getElementById('authError').textContent='';
}

async function api(url,options={}){
 const response=await fetch(url,{
  credentials:'same-origin',
  ...options,
  headers:{'Content-Type':'application/json',...(options.headers||{})}
 });
 const data=await response.json().catch(()=>({detail:'Server xatosi'}));
 if(!response.ok)throw new Error(data.detail||'Server xatosi');
 return data;
}

async function authenticate(){
 const error=document.getElementById('authError');
 error.textContent='';

 try{
  const data={
   name:document.getElementById('name').value,
   email:document.getElementById('email').value,
   password:document.getElementById('password').value
  };

  await api(authMode==='login'?'/api/login':'/api/register',{
   method:'POST',body:JSON.stringify(data)
  });

  await boot();
 }catch(e){
  error.textContent=e.message;
 }
}

async function boot(){
 try{
  const user=await api('/api/me');

  document.getElementById('auth').classList.add('hidden');
  document.getElementById('application').classList.remove('hidden');

  document.getElementById('userName').textContent=user.name;
  document.getElementById('profileName').textContent=user.name;
  document.getElementById('profileEmail').textContent=user.email;

  await loadProjects();
 }catch(e){
  document.getElementById('application').classList.add('hidden');
  document.getElementById('auth').classList.remove('hidden');
 }
}

async function logout(){
 await api('/api/logout',{method:'POST'});
 location.reload();
}

function page(name,button){
 ['studio','projects','account','about'].forEach(id=>{
  document.getElementById(id).classList.toggle('hidden',id!==name);
 });

 document.querySelectorAll('.side-button').forEach(x=>x.classList.remove('active'));
 if(button)button.classList.add('active');

 if(name==='projects')loadProjects();
}

async function createProject(){
 const status=document.getElementById('jobStatus');
 const prompts=document.getElementById('prompts').value;

 if(!prompts.trim()){
  status.textContent='Prompt yozing.';
  return;
 }

 status.textContent='Loyiha navbatga yuborilmoqda...';

 try{
  const job=await api('/api/projects',{
   method:'POST',
   body:JSON.stringify({
    title:document.getElementById('title').value,
    prompts:prompts,
    style:document.getElementById('style').value,
    duration:Number(document.getElementById('duration').value)
   })
  });

  status.textContent='Navbatda. Loyiha ID: '+job.id;

  if(pollTimer)clearInterval(pollTimer);

  pollTimer=setInterval(()=>watchJob(job.id),5000);
  await loadProjects();

 }catch(e){
  status.textContent=e.message;
 }
}

async function watchJob(id){
 try{
  const job=await api('/api/projects/'+id);
  const status=document.getElementById('jobStatus');

  status.textContent='Holat: '+job.status+
   (job.error?' — '+job.error:'');

  if(job.status==='completed'){
   clearInterval(pollTimer);

   const preview=document.getElementById('preview');
   preview.innerHTML='';

   const video=document.createElement('video');
   video.controls=true;
   video.playsInline=true;
   video.src=job.video;
   preview.appendChild(video);

   const link=document.createElement('a');
   link.href=job.video;
   link.download='vyra-video.mp4';
   link.textContent='↓ Videoni yuklab olish';
   link.style.display='inline-block';
   link.style.marginTop='15px';

   preview.after(link);
   await loadProjects();
  }

  if(job.status==='failed'){
   clearInterval(pollTimer);
   await loadProjects();
  }

 }catch(e){
  document.getElementById('jobStatus').textContent=e.message;
 }
}

async function loadProjects(){
 try{
  const data=await api('/api/projects');
  document.getElementById('count').textContent=data.length;

  const list=document.getElementById('projectList');
  list.innerHTML='';

  if(!data.length){
   list.textContent='Hali loyihalar yo‘q.';
   return;
  }

  data.forEach(item=>{
   const div=document.createElement('div');
   div.className='project';

   const info=document.createElement('div');
   const title=document.createElement('b');
   title.textContent=item.title;

   const sub=document.createElement('div');
   sub.className='status';
   sub.textContent=item.status+' · '+item.created;

   info.append(title,sub);
   div.append(info);

   if(item.video){
    const a=document.createElement('a');
    a.href=item.video;
    a.textContent='Videoni ochish';
    div.append(a);
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
# PUBLIC ROUTES
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "app": APP_NAME,
        "gpu_worker": "separate Colab runtime"
    }


# =========================================================
# AUTH ROUTES
# =========================================================

@app.post("/api/register")
async def register(request: Request):
    data = await request.json()

    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not 2 <= len(name) <= 60:
        raise HTTPException(400, "Ism 2-60 belgi bo‘lishi kerak.")

    if len(email) > 150 or not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+", email
    ):
        raise HTTPException(400, "Email noto‘g‘ri.")

    if not 8 <= len(password) <= 128:
        raise HTTPException(400, "Parol 8-128 belgi bo‘lishi kerak.")

    with database() as db:
        try:
            cursor = db.execute(
                """
                INSERT INTO users(name,email,password,created)
                VALUES(?,?,?,?)
                """,
                (name, email, password_hash(password), time.time())
            )
            user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Email allaqachon ro‘yxatdan o‘tgan.")

    request.session.clear()
    request.session["user_id"] = user_id

    return {"success": True}


@app.post("/api/login")
async def login(request: Request):
    data = await request.json()

    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    with database() as db:
        user = db.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        ).fetchone()

    if not user or not password_verify(password, user["password"]):
        raise HTTPException(401, "Email yoki parol noto‘g‘ri.")

    request.session.clear()
    request.session["user_id"] = user["id"]

    return {"success": True}


@app.post("/api/logout")
async def logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/me")
async def me(request: Request):
    user = auth_required(request)

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"]
    }


# =========================================================
# PROJECT ROUTES
# =========================================================

@app.post("/api/projects")
async def create_project(request: Request):
    user = auth_required(request)
    data = await request.json()

    title = str(data.get("title", "My project")).strip()[:100]
    raw_prompts = str(data.get("prompts", "")).strip()
    style = str(data.get("style", "Cinematic"))
    duration = data.get("duration", 2)

    if not raw_prompts or len(raw_prompts) > MAX_PROMPT_LENGTH:
        raise HTTPException(400, "Prompt 1-5000 belgi bo‘lishi kerak.")

    prompts = [
        line.strip()
        for line in raw_prompts.splitlines()
        if line.strip()
    ]

    if not prompts or len(prompts) > MAX_SCENES:
        raise HTTPException(400, "1 dan 3 tagacha sahna kiriting.")

    if style not in STYLES:
        raise HTTPException(400, "Video uslubi noto‘g‘ri.")

    if type(duration) is not int or duration not in (2, 3, 4, 5):
        raise HTTPException(400, "Sahna davomiyligi noto‘g‘ri.")

    project_id = uuid.uuid4().hex
    now = time.time()

    with database() as db:
        db.execute(
            """
            INSERT INTO projects
            (id,user_id,title,prompts,style,duration,status,created,updated)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                project_id,
                user["id"],
                title or "My project",
                json.dumps(prompts, ensure_ascii=False),
                style,
                duration,
                "queued",
                now,
                now
            )
        )

    return {
        "id": project_id,
        "status": "queued"
    }


@app.get("/api/projects")
async def list_projects(request: Request):
    user = auth_required(request)

    with database() as db:
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
            "created": datetime.fromtimestamp(
                row["created"]
            ).strftime("%Y-%m-%d %H:%M"),
            "video": (
                f"/api/videos/{row['id']}"
                if row["video"] else None
            )
        }
        for row in rows
    ]


@app.get("/api/projects/{project_id}")
async def project_status(project_id: str, request: Request):
    user = auth_required(request)

    with database() as db:
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
            f"/api/videos/{project_id}"
            if row["video"] else None
        )
    }


# =========================================================
# COLAB WORKER API
# =========================================================

@app.get("/api/worker/status")
async def worker_status(request: Request):
    check_worker(request)

    with database() as db:
        queued = db.execute(
            "SELECT COUNT(*) FROM projects WHERE status='queued'"
        ).fetchone()[0]

        processing = db.execute(
            "SELECT COUNT(*) FROM projects WHERE status='processing'"
        ).fetchone()[0]

    return {
        "status": "connected",
        "queued": queued,
        "processing": processing
    }


@app.get("/api/worker/claim")
async def worker_claim(request: Request):
    check_worker(request)

    now = time.time()

    with database() as db:
        # Worker uzilib qolgan bo‘lsa, eski vazifani qayta navbatga qo‘yish.
        db.execute(
            """
            UPDATE projects
            SET status='queued', updated=?
            WHERE status='processing' AND updated<?
            """,
            (now, now - 1800)
        )

        # BEGIN IMMEDIATE bir vaqtning o‘zida ikki worker
        # bitta topshiriqni olib ketishining oldini oladi.
        db.execute("BEGIN IMMEDIATE")

        row = db.execute(
            """
            SELECT *
            FROM projects
            WHERE status='queued'
            ORDER BY created ASC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            return {"job": None}

        db.execute(
            """
            UPDATE projects
            SET status='processing', updated=?
            WHERE id=? AND status='queued'
            """,
            (now, row["id"])
        )

        return {
            "job": {
                "id": row["id"],
                "title": row["title"],
                "prompts": json.loads(row["prompts"]),
                "style": row["style"],
                "duration": row["duration"]
            }
        }


@app.post("/api/worker/complete/{project_id}")
async def worker_complete(
    project_id: str,
    request: Request,
    video: UploadFile = File(...)
):
    check_worker(request)

    if video.content_type not in (
        "video/mp4",
        "application/octet-stream"
    ):
        raise HTTPException(400, "MP4 video yuboring.")

    with database() as db:
        row = db.execute(
            """
            SELECT id,status
            FROM projects
            WHERE id=?
            """,
            (project_id,)
        ).fetchone()

    if not row or row["status"] != "processing":
        raise HTTPException(404, "Faol topshiriq topilmadi.")

    filename = f"{project_id}.mp4"
    final_path = VIDEO_DIR / filename
    temp_path = VIDEO_DIR / f"{project_id}.upload"

    total = 0

    try:
        with open(temp_path, "wb") as output:
            while True:
                chunk = await video.read(1024 * 1024)

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_VIDEO_SIZE:
                    raise HTTPException(413, "Video hajmi 100 MB dan oshdi.")

                output.write(chunk)

        if total < 1024:
            raise HTTPException(400, "Video fayli bo‘sh.")

        with open(temp_path, "rb") as check:
            header = check.read(12)

        if len(header) < 8 or header[4:8] != b"ftyp":
            raise HTTPException(400, "Yuborilgan fayl haqiqiy MP4 emas.")

        os.replace(temp_path, final_path)

        with database() as db:
            db.execute(
                """
                UPDATE projects
                SET status='completed',video=?,error=NULL,updated=?
                WHERE id=?
                """,
                (filename, time.time(), project_id)
            )

        return {
            "success": True,
            "status": "completed",
            "video": filename
        }

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    finally:
        await video.close()


@app.post("/api/worker/fail/{project_id}")
async def worker_fail(project_id: str, request: Request):
    check_worker(request)

    data = await request.json()
    error = str(data.get("error", "GPU generation failed"))[:1000]

    with database() as db:
        cursor = db.execute(
            """
            UPDATE projects
            SET status='failed',error=?,updated=?
            WHERE id=? AND status='processing'
            """,
            (error, time.time(), project_id)
        )

        if cursor.rowcount == 0:
            raise HTTPException(404, "Topshiriq topilmadi.")

    return {"success": True}


# =========================================================
# VIDEO DOWNLOAD
# =========================================================

@app.get("/api/videos/{project_id}")
async def download_video(project_id: str, request: Request):
    user = auth_required(request)

    with database() as db:
        row = db.execute(
            """
            SELECT status,video
            FROM projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user["id"])
        ).fetchone()

    if not row or row["status"] != "completed" or not row["video"]:
        raise HTTPException(404, "Video tayyor emas.")

    filename = Path(row["video"]).name
    path = VIDEO_DIR / filename

    if not path.is_file():
        raise HTTPException(404, "Video fayli topilmadi.")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=filename
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/system")
async def system_status():
    return {
        "platform": APP_NAME,
        "backend": "online",
        "video_engine": "Wan 2.1 via Google Colab",
        "paid_ai_api": False
    }
