from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
import torch
import re

# Hugging Face login
login("#####")

# Cihaz ayarı (MPS = Mac GPU)
device = torch.device("mps")

# float32 precision ayarı
torch.set_float32_matmul_precision("high")

# FastAPI app
app = FastAPI()

# CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Girdi modeli
class JokeRequest(BaseModel):
    prompt: str

# Model ve tokenizer yolları
base_model_name = "ytu-ce-cosmos/turkish-gpt2-large"
adapter_path = "./GPT2"

# Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Model yükle ve MPS'e aktar
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32  # İstersen float16 da deneyebilirsin
).to(device)

# Fine-tuned adapteri yükle
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

# Temizleme fonksiyonu
def clean_joke_output(raw_text):
    content = raw_text.split("[/INST]")[-1]
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\[/?[^\]]+\]', '', content)
    content = re.sub(r'^\s*[\*\-\•]?\s*', '', content)
    content = re.sub(r'\b\d+\.\s*', '', content)
    content = content.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    content = content.replace("...", ".").replace("..", ".").replace("--", "-")
    content = re.sub(r'\s+', ' ', content)
    return content.strip()


# Uygun fıkra filtresi
def is_valid_joke(text):
    blacklist = [
         "http", "twitter", "facebook", "tweet", "www.", "youtube"
    ]
    return not any(word in text.lower() for word in blacklist) 

@app.post("/generate")
def generate_joke(data: JokeRequest):
    base_prompt = """<s>[INST] Aşağıdaki gibi kısa ve komik, günlük yaşamla ilgili fıkralar yazılmasını istiyorum. 
Lütfen sadece fıkra yaz, alıntı, yorum, analiz veya açıklama verme:

1. Temel sınavda kopya çekerken yakalanır. Öğretmen: 'Bu cevaplar aynı!' Temel: 'Aynı hocaya sorduk.'
2. Nasreddin Hoca bir gün göle yoğurt çalar. Komşusu sorar: 'Hoca ne yapıyorsun?' Hoca: 'Ya tutarsa!'
3. Temel arabayla giderken birden fren yapar. Dursun: 'Ne oldu?' Temel: 'Öndeki arabanın freni yokmuş, ben de saygı duydum.'
4. Dursun, Temel’e sorar: 'Sence kim daha zeki?' Temel: 'Hangimizin aklına bu soru geldi, o!'
5. Hoca pazardan limon alır. Satıcı: 'Tadı ekşi.' Hoca: 'Benim de yüzüm asık.'
6. Temel telefona sarılır: 'Alo, yangın var!' Görevli: 'Neresi yanıyor?' Temel: 'Telefonun diğer ucu!'
7. Hoca: 'Ben her şeye hazırlıklıyım.' Komşu: 'Depreme bile mi?' Hoca: 'Eşeğimi dışarıda tutuyorum zaten.'
8. Dursun: 'Temel, neden çalar saat almadın?' Temel: 'Uyanmak istemiyorum.'
9. Temel sınava boş kâğıt verir. Öğretmen: 'Bu ne?' Temel: 'Sessiz protesto.'
10. Hoca eşeğine ters binmiş. Soranlara: 'Ben değil, onlar ters gidiyor.' demiş.

Şimdi bunlara benzer kısa ve komik sonlu bir fıkra yazar mısın? Max 30 kelimede bitecek şekilde olmalı. [/INST]
"""

    for _ in range(5):
        inputs = tokenizer(base_prompt, return_tensors="pt").to(device)
        outputs = finetuned_model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.9,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        first_fikra = clean_joke_output(raw)
        if is_valid_joke(first_fikra):
            break
    else:
        return {"joke": "Uygun bir fıkra üretilemedi."}

    edit_prompt = f"""<s>[INST] Aşağıdaki fıkra olay akışı açısından dağınık ve yeterince komik değil.
Lütfen bu fıkrayı:
    Mantıklı bir olay sırasına oturt,
    Gereksiz tekrarları çıkar,
    Anlam bütünlüğünü koru,
    Kısa tut ve
    Esprili bir şekilde bitir.
Max 30 kelime olmalı ona dikkat et.

{first_fikra}

Düzenlenmiş fıkra: [/INST]"""

    inputs_edit = tokenizer(edit_prompt, return_tensors="pt").to(device)
    edited_output = base_model.generate(
        **inputs_edit,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.5,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.2,
        num_return_sequences=1
    )
    edited_text = tokenizer.decode(edited_output[0], skip_special_tokens=True)
    final_clean = clean_joke_output(edited_text)
    return {"joke": final_clean}