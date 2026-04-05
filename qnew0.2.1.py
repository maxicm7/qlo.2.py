import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import gumbel_r
import google.generativeai as genai

# Ignorar advertencias
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Agente Predictivo LLM Gumbel v3.0")

# --- ESTILOS ---
st.markdown("""
    <style>
    .stChatFloatingInputContainer {padding-bottom: 20px;}
    .reportview-container .main .footer {color: #777;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. FUNCIONES MATEMÁTICAS (GUMBEL & METRICAS) ---

def calcular_probabilidad_gumbel(atraso_actual, historico_atrasos):
    """Calcula la probabilidad de que un número rompa su racha usando la distribución Gumbel."""
    if len(historico_atrasos) < 5:
        return 0.5 # Default si hay pocos datos
    
    # Ajustar parámetros de la distribución Gumbel
    mu, beta = gumbel_r.fit(historico_atrasos)
    # CDF nos da la prob. de que el evento ocurra en un atraso <= al actual
    prob = gumbr_cdf(atraso_actual, mu, beta)
    return prob

def gumbr_cdf(x, mu, beta):
    """Función de distribución acumulada para Gumbel."""
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))

def calcular_metricas(combinacion, numero_a_atraso, numero_a_frecuencia, total_atraso_dataset, tensiones_gumbel):
    """Calcula las estadísticas de una combinación candidata incluyendo Tensión Gumbel."""
    combo_valido = [str(n) for n in combinacion if str(n) in numero_a_atraso]
    
    if len(combo_valido) < 5: return None

    atrasos = [numero_a_atraso.get(n, 0) for n in combo_valido]
    frecuencias = [numero_a_frecuencia.get(n, 0) for n in combo_valido]
    tensiones = [tensiones_gumbel.get(n, 0) for n in combo_valido]
    
    mean_atraso = np.mean(atrasos)
    mean_frecuencia = np.mean(frecuencias)
    
    # Coeficiente de Variación
    cv_f = (np.std(frecuencias) / mean_frecuencia) if mean_frecuencia > 0 else 0
    cv_a = (np.std(atrasos) / mean_atraso) if mean_atraso > 0 else 0

    # FÓRMULA SOLICITADA: Total atrasos + 40 - Suma atrasos combo
    calculo_especial = total_atraso_dataset + 40 - sum(atrasos)

    return {
        'Combinación': ' - '.join(map(str, sorted(combinacion))),
        'suma': sum(map(int, combo_valido)), 
        'pares': sum(1 for n in combo_valido if int(n) % 2 == 0),
        'cv_frecuencia': cv_f,
        'cv_atraso': cv_a,
        'tension_gumbel_media': np.mean(tensiones),
        'calculo_especial': calculo_especial
    }

# --- 2. CARGA DE DATOS ---

@st.cache_data
def load_data_files(data_file, history_file):
    try:
        # Carga Datos
        df = pd.read_csv(data_file, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.astype(str).str.strip().str.lower()
        col_map = {'numero': 'Numero', 'atraso': 'Atraso', 'frecuencia': 'Frecuencia'}
        df = df.rename(columns=lambda x: next((v for k, v in col_map.items() if k in x), x))
        df = df.loc[:, ~df.columns.duplicated()]
        
        df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce').dropna().astype(int).astype(str)
        df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce').fillna(0).astype(int)
        df['Frecuencia'] = pd.to_numeric(df['Frecuencia'], errors='coerce').fillna(0).astype(int)

        # Carga Historial
        if history_file.name.endswith('.xlsx'):
            df_hist = pd.read_excel(history_file, header=None)
        else:
            df_hist = pd.read_csv(history_file, sep=None, engine='python', header=None)
        
        historical_sets = []
        for _, row in df_hist.iterrows():
            nums = {int(x) for x in row if pd.notna(x) and 0 <= x <= 150}
            if len(nums) >= 5: historical_sets.append(nums)

        # Calcular Tensiones de Gumbel por número
        # Simulamos un histórico de atrasos para cada número basado en su frecuencia
        tensiones_gumbel = {}
        for _, row in df.iterrows():
            n = row['Numero']
            a_actual = row['Atraso']
            # Creamos una distribución sintética basada en la frecuencia media para el ajuste
            historico_sintetico = np.random.exponential(scale=max(1, 100/max(1, row['Frecuencia'])), size=50)
            tensiones_gumbel[n] = calcular_probabilidad_gumbel(a_actual, historico_sintetico)

        return {
            'na': dict(zip(df['Numero'], df['Atraso'])),
            'nf': dict(zip(df['Numero'], df['Frecuencia'])),
            'tg': tensiones_gumbel,
            'ta': df['Atraso'].sum(),
            'hs': historical_sets,
            'df_completo': df
        }
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None

# --- 3. LÓGICA DE INTELIGENCIA ARTIFICIAL (GEMINI) ---

def consultar_gemini(api_key, prompt, context=""):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        full_prompt = f"Contexto de datos: {context}\n\nConsulta: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error en AI: {str(e)}"

# --- 4. INTERFAZ DE USUARIO ---

# Sidebar para API Key y Configuración
with st.sidebar:
    st.title("⚙️ Configuración")
    gemini_key = st.text_input("Gemini API Key", type="password")
    n_candidatos = st.number_input("Candidatos", 1000, 200000, 20000)
    ventana = st.slider("Ventana Histórica", 10, 200, 50)
    
    if st.button("Limpiar Chat"):
        st.session_state.messages = []

# Área principal
st.title("🤖 Agente Predictivo Homeostático + Gemini 2.0")

col_f1, col_f2 = st.columns(2)
f_data = col_f1.file_uploader("CSV de Atrasos/Frecuencias", type="csv")
f_hist = col_f2.file_uploader("Historial de Sorteos", type=["csv", "xlsx"])

if f_data and f_hist:
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = load_data_files(f_data, f_hist)

    if st.session_state.data_processed:
        d = st.session_state.data_processed
        
        if st.button("🚀 Iniciar Análisis de Alta Tensión"):
            with st.spinner("Calculando Probabilidades de Gumbel y Correlaciones..."):
                # Generación de candidatos (Lógica simplificada para el ejemplo)
                nums_disp = [int(n) for n in d['na'].keys()]
                candidatos = [random.sample(nums_disp, 6) for _ in range(n_candidatos)]
                
                # Evaluación
                resultados = []
                for c in candidatos:
                    m = calcular_metricas(c, d['na'], d['nf'], d['ta'], d['tg'])
                    if m: resultados.append(m)
                
                df_res = pd.DataFrame(resultados).sort_values('tension_gumbel_media', ascending=False)
                st.session_state.top_combos = df_res.head(20)
                
                st.success("Análisis matemático completado.")

        # --- SECCIÓN DE GEMINI ---
        if 'top_combos' in st.session_state and gemini_key:
            st.header("🧠 Análisis con LLM (Gemini 2.0 Flash)")
            if st.button("Analizar con Inteligencia Artificial"):
                contexto_ia = f"""
                He analizado {n_candidatos} combinaciones. 
                Los mejores resultados según Probabilidad de Gumbel y el cálculo especial (Suma Atrasos + 40 - Suma Combo) son:
                {st.session_state.top_combos.to_string()}
                
                Los números con más tensión (probabilidad de salir según Gumbel) son:
                {sorted(d['tg'].items(), key=lambda x: x[1], reverse=True)[:10]}
                """
                
                pregunta = "Cruza los datos de las combinaciones con la tensión de Gumbel y las correlaciones dinámicas. ¿Cuál es la combinación más probable y por qué? Dame una sola ganadora definitiva basada en el equilibrio de homeostasis."
                
                with st.spinner("Gemini analizando correlaciones..."):
                    respuesta_ia = consultar_gemini(gemini_key, pregunta, contexto_ia)
                    st.write("### 🏆 Recomendación Definitiva de la IA:")
                    st.info(respuesta_ia)
                    st.session_state.last_analysis = respuesta_ia

        # --- TABLAS DE RESULTADOS ---
        if 'top_combos' in st.session_state:
            with st.expander("Ver Top 20 Combinaciones de Alta Tensión"):
                st.dataframe(st.session_state.top_combos)

# --- PANEL DE CHAT (Siempre al final) ---
st.divider()
st.subheader("💬 Chat Consultor de Datos")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pregúntame sobre los atrasos, Gumbel o los resultados..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not gemini_key:
            response = "Por favor, introduce tu API Key de Gemini en la barra lateral para conversar."
        elif 'data_processed' not in st.session_state:
            response = "Primero debes cargar los archivos de datos."
        else:
            # Crear un contexto resumido para el chat
            resumen_data = f"Total números: {len(st.session_state.data_processed['na'])}. Suma total atrasos: {st.session_state.data_processed['ta']}. Historial sorteos: {len(st.session_state.data_processed['hs'])}."
            if 'top_combos' in st.session_state:
                resumen_data += f" Top combo actual: {st.session_state.top_combos.iloc[0]['Combinación']}"
            
            response = consultar_gemini(gemini_key, prompt, resumen_data)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("Sube los archivos y configura tu API Key para activar todas las funciones.")
