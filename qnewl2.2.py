import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import warnings
from scipy.stats import gumbel_r
import google.generativeai as genai

warnings.filterwarnings("ignore")

# --- 1. CARGA DE DATOS (Diagrama Bloque 1 - CORREGIDO) ---
@st.cache_data
def load_and_process_data(f_data, f_hist):
    # Lectura robusta de CSV
    try:
        df = pd.read_csv(f_data, sep=None, engine='python')
    except:
        f_data.seek(0)
        df = pd.read_csv(f_data, sep=';')

    # CORRECCIÓN DE ERROR: Aplicar str a cada paso
    df.columns = df.columns.astype(str).str.strip().str.capitalize()
    
    # Procesar Historial
    if f_hist.name.endswith('.xlsx'):
        df_h = pd.read_excel(f_hist, header=None)
    else:
        df_h = pd.read_csv(f_hist, header=None, sep=None, engine='python')
    
    historial_sets = [set(row.dropna().astype(int)) for _, row in df_h.iterrows()]
    
    # Mapeos rápidos
    # Buscamos columnas Numero/Atraso/Frecuencia aunque varíe la mayúscula
    col_n = [c for c in df.columns if 'Num' in c][0]
    col_a = [c for c in df.columns if 'Atra' in c][0]
    col_f = [c for c in df.columns if 'Frec' in c][0]

    atraso_map = dict(zip(df[col_n].astype(int), df[col_a].astype(int)))
    frec_map = dict(zip(df[col_n].astype(int), df[col_f].astype(int)))
    total_atraso_dataset = df[col_a].sum()
    
    return df, historial_sets, atraso_map, frec_map, total_atraso_dataset, col_a

# --- 2. CÁLCULO GUMBEL (Diagrama Bloque 2) ---
def get_gumbel_tensions(delays_series, atraso_map):
    mu, beta = gumbel_r.fit(delays_series)
    # Tensión = Probabilidad de que el atraso actual sea un evento extremo
    gumbel_map = {n: gumbel_r.cdf(a, loc=mu, scale=beta) for n, a in atraso_map.items()}
    return gumbel_map, mu, beta

# --- 3. HOMEOSTASIS GLOBAL (Diagrama Bloque 3) ---
def calcular_reglas_homeostaticas(historial_sets, atraso_map, frec_map):
    metricas = []
    for s in historial_sets:
        nums = [n for n in s if n in atraso_map]
        if len(nums) < 5: continue
        atrasos = [atraso_map[n] for n in nums]
        frecuencias = [frec_map[n] for n in nums]
        metricas.append({
            'suma': sum(nums),
            'cv_a': np.std(atrasos) / np.mean(atrasos) if np.mean(atrasos) > 0 else 0,
            'pares': sum(1 for n in nums if n % 2 == 0)
        })
    
    df_m = pd.DataFrame(metricas)
    reglas = {
        'suma': (df_m['suma'].mean() - 2.5 * df_m['suma'].std(), df_m['suma'].mean() + 2.5 * df_m['suma'].std()),
        'cv_a': (df_m['cv_a'].mean() - 2.5 * df_m['cv_a'].std(), df_m['cv_a'].mean() + 2.5 * df_m['cv_a'].std()),
        'pares': set(df_m['pares'].unique())
    }
    return reglas

# --- 4. CORRELACIÓN DINÁMICA (Diagrama Bloque 4) ---
def get_dynamic_correlation(historial_sets, window=50):
    recent = historial_sets[-window:]
    corr_matrix = np.zeros((151, 151))
    for s in recent:
        l = sorted(list(s))
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                n1, n2 = int(l[i]), int(l[j])
                if n1 <= 150 and n2 <= 150:
                    corr_matrix[n1][n2] += 1
                    corr_matrix[n2][n1] += 1
    return corr_matrix

# --- 5, 6 y 7. GENERACIÓN, FILTRADO Y SCORING (500k) ---
def motor_500k(n_combos, nums_disp, atraso_map, gumbel_map, corr_matrix, reglas, total_atraso):
    candidatos = []
    nums_array = np.array(nums_disp)
    batch_size = 50000
    
    for _ in range(n_combos // batch_size):
        # Bloque 5: Generación
        batch = np.array([np.random.choice(nums_array, 6, replace=False) for _ in range(batch_size)])
        
        # Bloque 6: Filtrado Homeostático (Suma)
        sumas = batch.sum(axis=1)
        mask = (sumas >= reglas['suma'][0]) & (sumas <= reglas['suma'][1])
        batch = batch[mask]
        
        # Bloque 7: Scoring
        for combo in batch:
            c_atrasos = [atraso_map[n] for n in combo]
            # FÓRMULA ESPECIAL: (Total + 40) - SumaCombo
            calc_especial = (total_atraso + 40) - sum(c_atrasos)
            
            tension = np.mean([gumbel_map[n] for n in combo])
            
            # Correlación (Socios)
            score_corr = sum(corr_matrix[combo[i]][combo[j]] for i in range(6) for j in range(i+1, 6))
            
            # Score Final (Ponderado)
            score_final = (tension * 70) + (score_corr * 10) + (calc_especial / 1000)
            
            candidatos.append({
                'Combinación': sorted(combo.tolist()),
                'Tension_Gumbel': round(tension, 4),
                'Correlacion': score_corr,
                'Calculo_Especial': calc_especial,
                'Score_Final': score_final
            })
            
    return pd.DataFrame(candidatos).sort_values('Score_Final', ascending=False)

# --- INTERFAZ STREAMLIT ---
st.set_page_config(layout="wide", page_title="Predictor Gumbel 500k")

# Barra lateral con persistencia (Usando keys)
with st.sidebar:
    st.header("⚙️ Ajustes")
    api_key = st.text_input("Gemini API Key", type="password", key="api_key")
    n_generar = st.select_slider("Candidatos a evaluar", options=[10000, 100000, 500000], value=500000, key="n_generar")
    ventana = st.slider("Ventana Correlación", 10, 200, 50, key="ventana")
    st.divider()
    if st.button("Limpiar Sesión"):
        st.cache_data.clear()
        st.rerun()

st.title("🤖 Predictor Homeostático v4.6")

c1, c2 = st.columns(2)
f_data = c1.file_uploader("Subir Atrasos (CSV)", type="csv")
f_hist = c2.file_uploader("Subir Historial (CSV/XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    # 1. Pipeline de Datos
    df_raw, historial, na, nf, ta, col_atraso = load_and_process_data(f_data, f_hist)
    
    # 2. Gumbel
    tg, mu_g, beta_g = get_gumbel_tensions(df_raw[col_atraso], na)
    
    # 3. Homeostasis
    reglas = calcular_reglas_homeostaticas(historial, na, nf)
    
    # 4. Correlación
    corr_matrix = get_dynamic_correlation(historial, ventana)
    
    if st.button(f"🔥 Iniciar Análisis Masivo ({n_generar:,})"):
        start = time.time()
        with st.spinner("Procesando 500,000 combinaciones..."):
            df_final = motor_500k(n_generar, list(na.keys()), na, tg, corr_matrix, reglas, ta)
            st.session_state.df_final = df_final
            st.success(f"Completado en {time.time()-start:.2f}s")
            
            st.subheader("🏆 Top de Combinaciones (Filtrado Homeostático + Gumbel)")
            st.dataframe(df_final.head(30), use_container_width=True)

    # 8. Análisis Gemini
    if api_key and 'df_final' in st.session_state:
        st.divider()
        if st.button("🧠 Consultar Veredicto Gemini 2.0 Flash"):
            top_context = st.session_state.df_final.head(20).to_string()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            Basado en el proceso: Gumbel, Homeostasis Global y Correlación Dinámica.
            Datos Top 20: {top_context}
            Atraso Total: {ta}. Parámetros Gumbel: mu={mu_g}, beta={beta_g}.
            
            Instrucción: Identifica la combinación ganadora justificando el cruce entre 'Calculo_Especial' y la 'Tensión de Gumbel'.
            """
            with st.spinner("Analizando con IA..."):
                res = model.generate_content(prompt)
                st.info(res.text)

# Chat persistente
if 'df_final' in st.session_state:
    st.divider()
    st.subheader("💬 Chat Consultor")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    if p := st.chat_input("Consulta sobre atrasos o tensiones..."):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        with st.chat_message("assistant"):
            if api_key:
                model = genai.GenerativeModel('gemini-2.0-flash')
                ctx = st.session_state.df_final.head(10).to_string()
                r = model.generate_content(f"Contexto: {ctx}. Pregunta: {p}")
                ans = r.text
            else: ans = "Introduce la API Key."
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
