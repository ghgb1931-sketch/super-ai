from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from groq import Groq
import google.generativeai as genai
import os, re, json, datetime, requests
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO

app = FastAPI()

# =====================
# إعداد المفاتيح
# =====================
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY", "")

# =====================
# الموديلات
# =====================
MODELS = {
    "general":     {"provider":"groq",   "model":"openai/gpt-oss-120b",                     "name":"👑 GPT-OSS 120B",      "desc":"الأقوى"},
    "reasoning":   {"provider":"groq",   "model":"qwen/qwen3-32b",                          "name":"🧮 Qwen3 32B",          "desc":"كود + رياضيات"},
    "powerful":    {"provider":"groq",   "model":"llama-3.3-70b-versatile",                 "name":"💪 Llama 3.3 70B",      "desc":"أسئلة صعبة"},
    "fast":        {"provider":"groq",   "model":"llama-3.1-8b-instant",                    "name":"⚡ Llama 3.1 8B",       "desc":"سريع"},
    "longcontext": {"provider":"groq",   "model":"meta-llama/llama-4-scout-17b-16e-instruct","name":"📄 Llama 4 Scout",      "desc":"سياق طويل"},
    "arabic":      {"provider":"groq",   "model":"allam-2-7b",                              "name":"🌍 Allam 2",             "desc":"عربي"},
    "websearch":   {"provider":"groq",   "model":"llama-3.3-70b-versatile",                 "name":"🔍 Llama Search",        "desc":"بحث"},
    "multimodal":  {"provider":"google", "model":"gemini-2.0-flash",                        "name":"✨ Gemini 2.0 Flash",    "desc":"صور"},
    "longdoc":     {"provider":"google", "model":"gemini-1.5-flash",                        "name":"📋 Gemini 1.5 Flash",   "desc":"ملفات"},
}

# =====================
# الذاكرة والمعرفة
# =====================
conversation_history = []
knowledge_base       = []

# =====================
# الـ Router
# =====================
def detect_domain(question):
    q = question.lower()
    if any(w in q for w in ["ابحث","بحث","اخبار","search","news","دلوقتي","الان","سعر"]):
        return "websearch"
    if any(w in q for w in ["كود","code","برمج","python","bug","error","sql","html","function"]):
        return "reasoning"
    if any(w in q for w in ["رياضيات","math","معادلة","حساب","فيزياء","علوم","احسب"]):
        return "reasoning"
    if any(w in q for w in ["اقرأ","حلل","analyze","استخرج","قارن","افهم"]):
        return "longcontext"
    if any(w in q for w in ["لخص","summarize","ملخص","كتاب","تقرير","pdf"]):
        return "longdoc"
    if any(w in q for w in ["صورة","image","photo","صوره"]):
        return "multimodal"
    if any(w in q for w in ["بالعربي","عربي","فصحى"]):
        return "arabic"
    if any(w in q for w in ["فلسفة","تاريخ","سياسة","لماذا","ازاي","كيف","اشرح","فرق"]):
        return "powerful"
    if any(w in q for w in ["اكتب","قصيدة","قصة","poem","story"]):
        return "general"
    if len(question) < 30:
        return "fast"
    return "general"

# =====================
# الـ RAG
# =====================
def chunk_text(text, chunk_size=500):
    sentences = re.split(r'[.!?\n]+', text)
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 30]

def search_knowledge(question, top_k=3):
    if not knowledge_base:
        return []
    docs       = knowledge_base + [question]
    vectorizer = TfidfVectorizer()
    tfidf      = vectorizer.fit_transform(docs)
    scores     = cosine_similarity(tfidf[-1], tfidf[:-1])[0]
    top_idx    = np.argsort(scores)[::-1][:top_k]
    return [knowledge_base[i] for i in top_idx if scores[i] > 0.05]

# =====================
# الـ Tools
# =====================
def tool_calculator(expr):
    try:
        expr   = expr.replace('%', '/100')
        result = eval(expr)
        return f"النتيجة: {result:,.6f}".rstrip('0').rstrip('.')
    except:
        return None

def tool_datetime():
    now = datetime.datetime.now()
    return f"التاريخ: {now.strftime('%Y-%m-%d')} | الوقت: {now.strftime('%H:%M:%S')}"

def detect_tool(question):
    q = question.lower()
    if any(w in q for w in ["احسب","حساب","كام","يساوي","ضرب","قسمة","جمع","طرح","فايدة"]):
        return "calculator"
    if any(w in q for w in ["تاريخ","وقت","date","time","النهارده","اليوم"]):
        return "datetime"
    return None

# =====================
# الاستجابة الرئيسية
# =====================
def get_answer(question):
    tool_name   = detect_tool(question)
    tool_result = None

    if tool_name == "calculator":
        nums = re.findall(r'[\d+\-*/().%\s]+', question)
        expr = max(nums, key=len).strip() if nums else ""
        if expr:
            tool_result = tool_calculator(expr)
    elif tool_name == "datetime":
        tool_result = tool_datetime()

    domain     = detect_domain(question)
    model_info = MODELS.get(domain, MODELS["general"])
    relevant   = search_knowledge(question)

    context = ""
    if relevant:
        context += "معلومات من الذاكرة:\n" + "\n".join(relevant) + "\n\n"
    if tool_result:
        context += f"نتيجة الحساب: {tool_result}\n\n"

    full_q = f"{context}السؤال: {question}" if context else question

    conversation_history.append({"role": "user", "content": full_q})

    try:
        if model_info["provider"] == "groq":
            client   = Groq(api_key=GROQ_KEY)
            messages = [
                {"role": "system", "content": "أنت مساعد ذكي ومفيد. أجب باللغة التي يكتب بها المستخدم."}
            ] + conversation_history[-10:]
            resp   = client.chat.completions.create(
                model=model_info["model"], messages=messages,
                max_tokens=2048, temperature=0.7
            )
            answer = resp.choices[0].message.content
        else:
            genai.configure(api_key=GOOGLE_KEY)
            model  = genai.GenerativeModel(model_info["model"])
            answer = model.generate_content(full_q).text
    except Exception as e:
        answer = f"❌ خطأ: {str(e)}"

    conversation_history.append({"role": "assistant", "content": answer})

    return {
        "answer":     answer,
        "model":      model_info["name"],
        "domain":     domain,
        "tool":       tool_result,
        "rag_used":   len(relevant) > 0,
        "rag_chunks": len(relevant),
    }

# =====================
# الـ API Routes
# =====================
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    return get_answer(req.message)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text    = ""
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    else:
        text = content.decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    knowledge_base.extend(chunks)
    return {"message": f"✅ تم رفع {len(chunks)} قطعة", "total": len(knowledge_base)}

@app.post("/clear")
def clear():
    conversation_history.clear()
    return {"message": "✅ تم مسح المحادثة"}

@app.get("/models")
def get_models():
    return [{"key": k, "name": v["name"], "desc": v["desc"]} for k, v in MODELS.items()]

@app.get("/", response_class=HTMLResponse)
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
        
