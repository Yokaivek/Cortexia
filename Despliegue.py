import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts import promp_fuerte



# ===========================
# CONFIGURACIÓN GENERAL
# ===========================
load_dotenv(".env.k")  
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) #Carga la clave de API desde el archivo .env.k

#Titulo del sistema 
st.set_page_config(
    page_title="Cortexia",
    page_icon="🧠",
    layout="wide",
)

# ===========================
# RUTAS Y CONSTANTES
# ===========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models_2" / "model_0.896.keras"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ===========================
# CARGA DEL MODELO (CACHE)
# ===========================
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ===========================
# PREPROCESAMIENTO
# ===========================
def imagenet_preprocess(image: Image.Image):
    image = image.resize(IMG_SIZE)
    x = np.array(image).astype("float32") / 255.0  # Normalización básica a [0,1]

    mean = np.array([0.485, 0.456, 0.406]) # Media de ImageNet
    std  = np.array([0.229, 0.224, 0.225]) # Desviación estándar de ImageNet

    x = (x - mean) / std
    x = np.expand_dims(x, axis=0) # Agrega dimensión de batch para que el modelo lo procese

    return x

# ===========================
# PREDICCIÓN
# ===========================
def predict(image: Image.Image):
    img_array = imagenet_preprocess(image)
    preds = model.predict(img_array, verbose=0)[0]  # softmax → ya suman 1

    probs_dict = {
        CLASS_NAMES[i]: float(preds[i])
        for i in range(len(CLASS_NAMES))
    }

    # 🔹 Probabilidad binaria principal
    prob_no_tumor = probs_dict["No Tumor"]
    prob_tumor = 1.0 - prob_no_tumor

    # 🔹 Subtipos tumorales
    tumor_breakdown = {
        k: v
        for k, v in probs_dict.items()
        if k != "No Tumor"
    }

    # Clase más probable (original)
    idx = np.argmax(preds)
    predicted_class = CLASS_NAMES[idx]
    confidence = float(preds[idx])

    return {
        "Predicción de clase": predicted_class,
        "Confianza": confidence,
        "Probabilidad_no_tumor": prob_no_tumor,
        "Probabilidad_tumor": prob_tumor,
        "Subtipos tumorales": tumor_breakdown,
        "Todas las probabilidades": probs_dict
    }

# ===========================
# ASISTENTE CLÍNICO
# ===========================
def clinical_assistant(contexto, user_message, history):
    messages = [
        {"role": "system", "content": promp_fuerte},
        {"role": "system", "content": contexto},
    ]

    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content


# ===========================
# ESTILOS
# ===========================
st.markdown("""
<style>
.stApp { background-color: #e8f1fb !important; }
.card {
    background-color: rgba(255,255,255,0.9);
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.10);
    border: 1px solid #d3e2f4;
    margin-top: 20px;
}
@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #0f172a !important;
        color: #e5e7eb !important;
    }

    .card {
        background-color: #1e293b !important;
        color: #e5e7eb !important;
        border: 1px solid #334155;
    }
}

.badge {
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
    display:inline-block;
}

/* Light mode */
@media (prefers-color-scheme: light) {
    .badge {
        background-color: #3b82f6;
        color: white;
    }
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .badge {
        background-color: #60a5fa;
        color: #0f172a;
    }
}

</style>
""", unsafe_allow_html=True)

# ===========================
# ENCABEZADO
# ===========================
st.markdown("<h1 style='text-align:center'> 🧠 Cortexia </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Clasificación de tumores cerebrales MRI y asistencia medica </p>", unsafe_allow_html=True)

# ===========================
# LAYOUT
# ===========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Cargar imagen MRI T1")

    uploaded_file = st.file_uploader(
        "Formatos: JPG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Imagen cargada", width=420)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Diagnóstico")

    if st.button("Realizar diagnóstico", use_container_width=True):
        if not uploaded_file:
            st.warning("Sube una imagen primero.")
        else:
            with st.spinner("Analizando imagen..."):
                resultado = predict(img)

                st.session_state["diagnostico"] = resultado


                st.session_state["chat_history"] = []

    if "diagnostico" in st.session_state:

        diag = st.session_state["diagnostico"]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Diagnóstico")

        st.subheader("🧠 Evaluación primaria: Tumor vs No Tumor")

        if diag["Probabilidad_tumor"] > diag["Probabilidad_no_tumor"]:
            st.error(f"Probabilidad de TUMOR: {diag['Probabilidad_tumor']*100:.2f}%")
        else:
            st.success(f"Probabilidad de NO TUMOR: {diag['Probabilidad_no_tumor']*100:.2f}%")

        st.markdown("---")
        if diag["Probabilidad_tumor"] > 0.10:

            st.subheader("🔬 Posible subtipo tumoral")

            df = (
                pd.DataFrame(
                    [(k, v*100) for k, v in diag["Subtipos tumorales"].items()],
                    columns=["Tipo de tumor", "Probabilidad (%)"]
                )
                .sort_values("Probabilidad (%)", ascending=False)
                .reset_index(drop=True)
            )

            st.table(df)
            st.bar_chart(df.set_index("Tipo de tumor"))


        st.markdown("</div>", unsafe_allow_html=True)


if not uploaded_file:
    st.info("🔽 Sube una imagen MRI T1 para comenzar el análisis.")

if "diagnostico" in st.session_state:

        diag = st.session_state["diagnostico"]

        contexto_modelo = f"""
        Resultado del modelo de clasificación de tumores cerebrales (CNN):
        Probabilidad de tumor: {diag['Probabilidad_tumor']*100:.2f}%
        Probabilidad de no tumor: {diag['Probabilidad_no_tumor']*100:.2f}%

        Subtipo tumoral más probable: {diag['Predicción de clase']}
        Confianza del subtipo: {diag['Confianza']*100:.2f}%


        Probabilidades por clase:
        """ + "\n".join(
                [f"- {k}: {v*100:.2f}%" for k, v in diag["Todas las probabilidades"].items()
]
            )
        st.markdown("---")
        st.markdown("## 🩺 Asistente clínico de apoyo diagnóstico")

        st.markdown(
            "<div class='card'>"
            "<p><b>Este asistente interpreta los resultados del modelo y apoya "
            "la toma de decisiones clínicas. No sustituye el juicio médico.</b></p>"
            "</div>",
            unsafe_allow_html=True
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if not st.session_state["chat_history"]:
            respuesta_inicial = clinical_assistant(
                contexto=contexto_modelo,
                user_message="Interpreta los resultados y proporciona recomendaciones clínicas iniciales.",
                history=[]
            )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": respuesta_inicial}
            )
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "assistant":
                st.markdown(f"<div class='card'><p>{msg['content']}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align:right;'><p><b>Usuario:</b> {msg['content']}</p></div>", unsafe_allow_html=True)
        user_input = st.chat_input(
            "Haz una pregunta o solicita recomendaciones adicionales"
        )

        if user_input:
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Analizando..."):
                respuesta = clinical_assistant(
                    contexto=contexto_modelo,
                    user_message=user_input,
                    history=st.session_state["chat_history"][:-1]
                )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": respuesta}
            )

            st.rerun()
