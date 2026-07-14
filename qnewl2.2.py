import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import warnings
from scipy.stats import gumbel_r
import google.generativeai as genai

warnings.filterwarnings("ignore")

# ==============================================================================
# --- 1. FUNCIONES DE PROCESAMIENTO ---
# ==============================================================================
@st.cache_data
def load_and_process_data(f_data, f_hist):
    try:
        try:
            df = pd.read_csv(f_data, sep=None, engine='python')
        except:
            f_data.seek(0)
            df = pd.read_csv(f_data, sep=';')

        df.columns = df.columns.astype(str).str.strip().str.capitalize()
        col_n = next((c for c in df.columns if 'Num' in c), df.columns[0])
        col_a = next((c for c in df.columns if 'Atra' in c), df.columns[1])

        df[col_n] = pd.to_numeric(df[col_n], errors='coerce').fillna(0).astype(int)
        df[col_a] = pd.to_numeric(df[col_a], errors='coerce').fillna(0).astype(int)

        atraso_map = dict(zip(df[col_n], df[col_a]))
        total_atraso_dataset = df[col_a].sum()

        if f_hist.name.endswith('.xlsx'):
            df_h = pd.read_excel(f_hist, header=None)
        else:
            df_h = pd.read_csv(f_hist, header=None, sep=None, engine='python')

        historial_sets = []
        for _, row in df_h.iterrows():
            linea = pd.to_numeric(row, errors='coerce').dropna()
            validos = {int(x) for x in linea if 0 <= x <= 150}
            if len(validos) >= 5:
                historial_sets.append(validos)
        
        return df, historial_sets, atraso_map, total_atraso_dataset, col_a, col_n
    except Exception as e:
        st.error(f"Error en carga: {e}")
        return None

def get_gumbel_tensions(delays_series, atraso_map):
    mu, beta = gumbel_r.fit(delays_series)
    gumbel_map = {n: gumbel_r.cdf(a, loc=mu, scale=beta) for n, a in atraso_map.items()}
    return gumbel_map, mu, beta

def calcular_reglas_homeostaticas(historial_sets, atraso_map):
    metricas = [{'suma': sum(n for n in s if n in atraso_map)} for s in historial_sets]
    df_m = pd.DataFrame(metricas)
    return {'suma': (df_m['suma'].mean() - 3.2 * df_m['suma'].std(), df_m['suma'].mean() + 3.2 * df_m['suma'].std())}

def get_dynamic_correlation(historial_sets, window):
    recent = historial_sets[-window:] if len(historial_sets) > window else historial_sets
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

def motor_500k_v48(n_combos, nums_disp, atraso_map, gumbel_map, corr_matrix, reglas, total_atraso, df_raw, col_a, col_n, peso_formula):
    candidatos = []
    nums_array = np.array(nums_disp)
    calientes = set(df_raw[df_raw[col_a] <= 2][col_n].tolist())
    divisor_dinamico = max(100, total_atraso / 1.5)
    multiplicador_usuario = peso_formula / 100.0
    
    batch_size = 50000
    for _ in range(n_combos // batch_size):
        batch = np.array([np.random.choice(nums_array, 6, replace=False) for _ in range(batch_size)])
        sumas = batch.sum(axis=1)
        mask = (sumas >= reglas['suma'][0]) & (sumas <= reglas['suma'][1])
        batch = batch[mask]
        
        for combo in batch:
            combo_set = set(combo)
            atrasos_c = [atraso_map[n] for n in combo]
            tensiones_g = [gumbel_map[n] for n in combo]
            
            mean_tension, std_tension = np.mean(tensiones_g), np.std(tensiones_g)
            calc_especial = (total_atraso + 40) - sum(atrasos_c)
            valor_norm = (calc_especial / divisor_dinamico) * multiplicador_usuario
            corr = sum(corr_matrix[combo[i]][combo[j]] for i in range(6) for j in range(i+1, 6))
            n_calientes = len(combo_set.intersection(calientes))
            
            score = (mean_tension * 50) + (std_tension * 20) + (corr * 15) + (n_calientes * 10) + (valor_norm * 10)
            
            candidatos.append({
                'Combinación': sorted(combo.tolist()),
                'Tension_Gumbel': round(mean_tension, 4), 
                'Score_IA': score
            })
    return pd.DataFrame(candidatos).sort_values('Score_IA', ascending=False)

# ==============================================================================
# --- INTERFAZ ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Agente Predictivo v4.8")

with st.sidebar:
    st.header("⚙️ Ajustes")
    api_key = st.text_input("Gemini API Key", type="password")
    n_generar = st.select_slider("Cantidad", options=[10000, 100000, 500000], value=100000)
    peso_formula = st.slider("Influencia Fórmula (%)", 0, 100, 20)
    modelo_seleccionado = st.selectbox("Modelo IA", ["gemini-2.0-flash", "gemini-1.5-pro"])

st.title("🤖 Agente Predictivo v4.8")

c1, c2 = st.columns(2)
f_data = c1.file_uploader("Subir Atrasos (CSV)", type="csv")
f_hist = c2.file_uploader("Subir Historial (CSV/XLSX)", type=["csv", "xlsx"])

if f_data and f_hist:
    res = load_and_process_data(f_data, f_hist)
    if res:
        df_raw, historial, na, ta, col_atraso, col_numero = res
        tg, _, _ = get_gumbel_tensions(df_raw[col_atraso], na)
        reglas = calcular_reglas_homeostaticas(historial, na)
        corr_matrix = get_dynamic_correlation(historial, 80)
        
        if st.button("🔥 Ejecutar Análisis"):
            df_final = motor_500k_v48(n_generar, list(na.keys()), na, tg, corr_matrix, reglas, ta, df_raw, col_atraso, col_numero, peso_formula)
            st.session_state.df_final = df_final
            st.rerun()

if 'df_final' in st.session_state:
    st.dataframe(st.session_state.df_final.head(40), use_container_width=True)
    st.download_button("📥 Descargar CSV Completo", st.session_state.df_final.to_csv(index=False).encode('utf-8'), "analisis.csv")

    if api_key:
        if st.button("🧠 Análisis IA"):
            try:
                genai.configure(api_key=api_key, transport="rest")
                model = genai.GenerativeModel(modelo_seleccionado)
                res = model.generate_content(f"Analiza: {st.session_state.df_final.head(10).to_string()}")
                st.info(res.text)
            except Exception as e:
                st.error(f"Error IA: {e}")
