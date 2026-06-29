import os
import json
import requests
from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PyPDF2 import PdfReader
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI(title="LocalOps-Brain Pro")

print("Embedding modeli yükleniyor... Lütfen bekleyin.")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Sistem hazır! Tarayıcıdan http://localhost:8000 adresine gidin.")

# Bellekte tutulacak küresel değişkenler
index = None
chunks_metadata = []
chat_history = []  # Sohbet geçmişini tutan liste

# ULTRA PREMIUM, MODAL AYARLI VE YENİ NESİL STREAMING SOHBET ARAYÜZÜ
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LocalOps-Brain — Kurumsal Zeka Panel</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card: #131926;
            --card-border: #222d42;
            --accent: #38bdf8;
            --text-main: #f8fafc;
            --text-muted: #64748b;
            --text-desc: #94a3b8;
        }
        * { box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; transition: all 0.2s ease; }
        body { background-color: var(--bg); color: var(--text-main); margin: 0; padding: 40px 20px; display: flex; align-items: center; justify-content: center; min-height: 100vh; position: relative; }
        .app-wrapper { width: 100%; max-width: 750px; }
        
        /* ÜST BAR VE AYAR BUTONU */
        .top-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .settings-btn { background: none; border: 1px solid var(--card-border); color: var(--text-desc); padding: 8px 14px; border-radius: 10px; cursor: pointer; font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .settings-btn:hover { border-color: var(--accent); color: var(--accent); }
        
        .brand-section { text-align: center; margin-bottom: 40px; }
        .brand-badge { background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); color: var(--accent); padding: 6px 16px; border-radius: 100px; font-size: 12px; font-weight: 700; letter-spacing: 0.05em; display: inline-block; margin-bottom: 16px; }
        h1 { font-size: 36px; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 12px 0; background: linear-gradient(135deg, #fff 0%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: var(--text-desc); font-size: 15px; }
        
        .card { background: var(--card); border: 1px solid var(--card-border); border-radius: 20px; padding: 32px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); margin-bottom: 24px; }
        .card-header { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 20px; }
        
        .file-box { position: relative; border: 2px dashed #2d3a54; border-radius: 14px; padding: 35px 20px; text-align: center; background: rgba(19, 25, 38, 0.5); cursor: pointer; }
        .file-box:hover { border-color: var(--accent); }
        .file-box input[type="file"] { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        .file-icon { font-size: 36px; margin-bottom: 10px; display: block; }
        .file-text { color: #cbd5e1; font-weight: 500; font-size: 14px; display: block; }
        .file-subtext { color: var(--text-muted); font-size: 12px; margin-top: 6px; display: block; }
        
        textarea { width: 100%; height: 90px; background: #0b0f19; border: 1px solid var(--card-border); border-radius: 14px; color: var(--text-main); padding: 16px; font-size: 15px; resize: none; outline: none; line-height: 1.5; }
        textarea:focus { border-color: var(--accent); }
        .btn { background: var(--text-main); color: var(--bg); border: none; padding: 14px 28px; border-radius: 12px; font-weight: 700; font-size: 15px; cursor: pointer; width: 100%; margin-top: 16px; }
        .btn:hover { background: #ffffff; transform: translateY(-1px); }
        
        #status, #chat-container { margin-top: 20px; display: none; }
        .status-loading { padding: 15px; border-radius: 14px; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); color: var(--accent); font-size: 14px; }
        .status-success { padding: 15px; border-radius: 14px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); color: #34d399; font-size: 14px; }
        
        /* STREAMING CHAT ALANI */
        .ai-response { background: #0b0f19; border: 1px solid var(--card-border); color: #e2e8f0; padding: 24px; border-radius: 14px; font-size: 15px; min-height: 60px; line-height: 1.6; }
        .ai-title { font-weight: 700; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }

        /* DINAMIK AYARLAR MODALI */
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(5, 8, 15, 0.85); backdrop-filter: blur(8px); display: flex; align-items: center; justify-content: center; z-index: 1000; visibility: hidden; opacity: 0; }
        .modal-overlay.active { visibility: visible; opacity: 1; }
        .modal { background: var(--card); border: 1px solid var(--card-border); width: 100%; max-width: 450px; padding: 32px; border-radius: 20px; box-shadow: 0 30px 60px rgba(0,0,0,0.8); }
        .modal-title { font-size: 18px; font-weight: 700; margin-bottom: 20px; color: var(--text-main); }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; font-size: 13px; color: var(--text-desc); margin-bottom: 6px; font-weight: 600; }
        .form-group input { width: 100%; background: #0b0f19; border: 1px solid var(--card-border); padding: 12px; border-radius: 10px; color: #fff; font-size: 14px; outline: none; }
        .form-group input:focus { border-color: var(--accent); }
        .modal-buttons { display: flex; gap: 12px; margin-top: 24px; }
        .btn-secondary { background: #222d42; color: #fff; border: none; padding: 12px; border-radius: 10px; cursor: pointer; font-weight: 600; flex: 1; }
        .btn-secondary:hover { background: #2d3c58; }
        .btn-primary { background: var(--accent); color: #000; border: none; padding: 12px; border-radius: 10px; cursor: pointer; font-weight: 700; flex: 1; }
        .btn-primary:hover { background: #0ea5e9; }
    </style>
</head>
<body>

    <div class="app-wrapper">
        <div class="top-bar">
            <div></div>
            <button class="settings-btn" onclick="toggleModal(true)">⚙ Sunucu Ayarları</button>
        </div>

        <div class="brand-section">
            <span class="brand-badge">PRO MULTI-FORMAT ENVIRONMENT</span>
            <h1>LocalOps-Brain v3</h1>
            <div class="subtitle">PDF, Excel ve TXT Destekli, Hafızalı Akıllı RAG Sistemi</div>
        </div>
        
        <div class="card">
            <div class="card-header">01 / Doküman Entegrasyonu</div>
            <div class="file-box">
                <input type="file" id="multiFile" accept=".pdf,.xlsx,.xls,.txt" onchange="updateFileName()"/>
                <span class="file-icon">📁</span>
                <span id="upload-label" class="file-text">Analiz edilecek dosyayı sürükleyin veya seçin</span>
                <span class="file-subtext">PDF, Excel veya TXT formatları desteklenir</span>
            </div>
            <button class="btn" onclick="uploadFile()">Sisteme Öğret →</button>
            <div id="status"></div>
        </div>

        <div class="card">
            <div class="card-header">02 / Akıllı Sorgu & Kelime Akışı</div>
            <textarea id="question" placeholder="Doküman içeriği veya önceki konuşmalar hakkında soru sorun..."></textarea>
            <button class="btn" onclick="askQuestion()">Sorguyu Çalıştır</button>
            
            <div id="chat-container">
                <div class="ai-title" style="margin-top: 20px;">✦ SİSTEM YANITI</div>
                <div id="stream-box" class="ai-response">Yanıt bekleniyor...</div>
            </div>
        </div>
    </div>

    <div class="modal-overlay" id="settingsModal">
        <div class="modal">
            <div class="modal-title">⚙️ Yerel Yapay Zeka Ayarları</div>
            <div class="form-group">
                <label>Ollama / LLM Sunucu Adresi</label>
                <input type="text" id="llmUrl" placeholder="Örn: http://192.168.1.119:11434">
            </div>
            <div class="form-group">
                <label>Model Adı</label>
                <input type="text" id="modelName" placeholder="Örn: qwen2.5">
            </div>
            <div class="modal-buttons">
                <button class="btn-secondary" onclick="toggleModal(false)">Vazgeç</button>
                <button class="btn-primary" onclick="saveSettings()">Ayarları Kaydet</button>
            </div>
        </div>
    </div>

    <script>
        // Sayfa açıldığında varsayılan ayarları veya kullanıcının kaydettiği ayarları yükle
        window.onload = function() {
            if(!localStorage.getItem('llmUrl')) {
                localStorage.setItem('llmUrl', 'http://192.168.1.119:11434');
                localStorage.setItem('modelName', 'qwen2.5');
            }
            document.getElementById('llmUrl').value = localStorage.getItem('llmUrl');
            document.getElementById('modelName').value = localStorage.getItem('modelName');
        };

        function toggleModal(show) {
            const modal = document.getElementById('settingsModal');
            if(show) modal.classList.add('active');
            else modal.classList.remove('active');
        }

        function saveSettings() {
            localStorage.setItem('llmUrl', document.getElementById('llmUrl').value.trim());
            localStorage.setItem('modelName', document.getElementById('modelName').value.trim());
            alert('Ayarlar başarıyla tarayıcı belleğine kaydedildi!');
            toggleModal(false);
        }

        function updateFileName() {
            const fileInput = document.getElementById('multiFile');
            const label = document.getElementById('upload-label');
            if(fileInput.files.length > 0) {
                label.innerHTML = `Seçilen Dosya: <span style='color:var(--accent); font-weight:600;'>\${fileInput.files[0].name}</span>`;
            }
        }

        async function uploadFile() {
            const fileInput = document.getElementById('multiFile');
            const statusDiv = document.getElementById('status');
            if(fileInput.files.length === 0) { alert('Lütfen önce bir dosya seçin!'); return; }
            
            statusDiv.style.display = 'block'; 
            statusDiv.className = 'status-loading'; 
            statusDiv.innerHTML = '✨ Doküman yapısı analiz ediliyor ve vektör tabanına işleniyor...';
            
            let formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
                let response = await fetch("/upload-file/", { method: "POST", body: formData });
                let data = await response.json();
                
                if(response.ok) {
                    statusDiv.className = 'status-success'; 
                    statusDiv.innerHTML = '🎉 ' + data.message;
                } else {
                    statusDiv.className = 'status-loading'; 
                    statusDiv.innerHTML = '❌ Hata: ' + data.error;
                }
            } catch (e) {
                statusDiv.className = 'status-loading';
                statusDiv.innerHTML = '❌ Sunucu bağlantı hatası.';
            }
        }

        async function askQuestion() {
            const qInput = document.getElementById('question').value;
            const chatContainer = document.getElementById('chat-container');
            const streamBox = document.getElementById('stream-box');
            
            if(!qInput.trim()) { alert('Lütfen bir soru yazın!'); return; }
            
            chatContainer.style.display = 'block';
            streamBox.innerHTML = '🤖 Bellek taranıyor, kelimeler akıtılıyor...';
            
            // Kullanıcının modalda kaydettiği ayarları header'dan backend'e uçuruyoruz
            const userLlmUrl = localStorage.getItem('llmUrl');
            const userModelName = localStorage.getItem('modelName');

            let formData = new FormData();
            formData.append("question", qInput);
            formData.append("llm_url", userLlmUrl);
            formData.append("model_name", userModelName);

            try {
                let response = await fetch("/ask-question-stream/", { method: "POST", body: formData });
                
                if(!response.ok) {
                    let errData = await response.json();
                    streamBox.innerHTML = '❌ Hata: ' + errData.error;
                    return;
                }

                // ChatGPT tarzı kelime akışı okuma mekanizması (Streaming Reader)
                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                streamBox.innerHTML = ""; // Kutuyu temizle

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    let chunk = decoder.decode(value, { stream: true });
                    // Gelen verideki satır atlamalarını HTML formatına çevirerek akıtıyoruz
                    streamBox.innerHTML += chunk.replace(/\\n/g, '<br>');
                }

            } catch (e) {
                streamBox.innerHTML = '❌ Bağlantı hatası veya zaman aşımı.';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return HTML_INTERFACE

# PDF, EXCEL VE TXT DESTEKLEYEN YENI DOSYA YÜKLEME METODU
@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    global index, chunks_metadata, chat_history
    try:
        filename = file.filename.lower()
        raw_text = ""
        
        # 1. FORMAT KONTROL: PDF OKUMA
        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text: raw_text += text + "\n"
                
        # 2. FORMAT KONTROL: EXCEL OKUMA
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(file.file)
            # Excel'in her satırını anlamlı birer metin cümlesine çeviriyoruz
            raw_text = df.to_string(index=False)
            
        # 3. FORMAT KONTROL: TXT OKUMA
        elif filename.endswith('.txt'):
            content = await file.read()
            raw_text = content.decode("utf-8")
            
        else:
            return JSONResponse(status_code=400, content={"error": "Desteklenmeyen dosya formatı. Sadece PDF, Excel veya TXT."})

        if not raw_text.strip():
            return JSONResponse(status_code=400, content={"error": "Dosya içeriği boş veya okunamaz durumda."})
        
        # Dokümanı anlamlı parçalara ayırma
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = text_splitter.split_text(raw_text)
        chunks_metadata = chunks
        chat_history = [] # Yeni dosya yüklenince eski sohbet hafızasını temizle
        
        # Vektörleme ve FAISS kaydı
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return {"message": f"Dosya başarıyla hafızaya alındı! ({len(chunks)} bilgi segmenti oluşturuldu)"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# DINAMIK AYARLI VE SÜREKLİ SOHBET AKIŞI SAĞLAYAN (STREAMING) METOD
@app.post("/ask-question-stream/")
async def ask_question_stream(
    question: str = Form(...), 
    llm_url: str = Form(...), 
    model_name: str = Form(...)
):
    global index, chunks_metadata, chat_history
    if index is None:
        return JSONResponse(status_code=400, content={"error": "Önce yukarıdan bir dosya yükle kanka."})
    
    try:
        # 1. RAG: Dokümandan ilgili metin parçalarını cımbızla çekme
        question_embedding = embedding_model.encode([question], convert_to_numpy=True)
        _, indices = index.search(question_embedding, k=2)
        
        relevant_contexts = [chunks_metadata[idx] for idx in indices[0] if idx < len(chunks_metadata) and idx != -1]
        context_text = "\n".join(relevant_contexts)
        
        # 2. SOHBET HAFIZASI (MEMORY) OLUŞTURMA
        # OpenAI standardında mesaj listesini inşa ediyoruz
        messages = [
            {"role": "system", "content": f"Sen kurumsal bir asistansın. Verilen doküman içeriğine dayanarak net ve Türkçe bir cevap üret.\n\n[DOKÜMAN İÇERİĞİ]:\n{context_text}"}
        ]
        
        # Eski konuşmaları listeye ekleyerek hafızayı LLM'e aktarıyoruz
        for past_q, past_a in chat_history[-3:]:  # Son 3 konuşmayı hatırlasın (hafıza şişmesin)
            messages.append({"role": "user", "content": past_q})
            messages.append({"role": "assistant", "content": past_a})
            
        # Güncel soruyu ekle
        messages.append({"role": "user", "content": question})
        
        # 3. KULLANICININ MODALDA YAZDIĞI DINAMIK IP ADRESINE GÖRE OLLAMA FORMATI
        final_url = f"{llm_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,  # Yapay zekadan kelime kelime akış istiyoruz
            "temperature": 0.3
        }
        
        # KELİMELERİ SUNUCUDAN ALIP ANLIK OLARAK TARAYICIYA FIRLATAN GENERATOR
        def llm_stream_generator():
            full_response_text = ""
            try:
                # Zaman aşımını (timeout) kapattık çünkü veri parça parça akacak
                response = requests.post(final_url, json=payload, stream=True, timeout=300)
                
                if response.status_code != 200:
                    yield f"[Hata]: Sunucu {response.status_code} döndürdü. IP veya Model adını kontrol edin."
                    return

                for line in response.iter_lines():
                    if line:
                        # Gelen satırı çöz ve 'data: ' ön ekini temizle
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_content = decoded_line[6:]
                            if data_content == "[DONE]": break
                            
                            try:
                                json_data = json.loads(data_content)
                                token = json_data["choices"][0]["delta"].get("content", "")
                                full_response_text += token
                                yield token # Kelimeyi anlık olarak web arayüzüne üfle
                            except:
                                pass
                                
                # Sohbet başarıyla bittiğinde bu cevabı hafızaya yazıyoruz
                chat_history.append((question, full_response_text))
                
            except Exception as e:
                # Sunucuya bağlanamazsa düşeceği akıllı B planı
                yield f"⚠️ [Yerel Yapay Zekaya Bağlanılamadı! (Ayarları Kontrol Edin)]\n\n[RAG Veritabanından Çıkarılan Ham Metin]:\n{context_text}"

        return StreamingResponse(llm_stream_generator(), media_type="text/event-stream")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)