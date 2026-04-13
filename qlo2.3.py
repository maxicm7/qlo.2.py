import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os
import google.generativeai as genai

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

st.set_page_config(page_title="PIV-60 v5.0 + Gemini", page_icon="🤖", layout="wide")

# =============================================================================
# CONFIGURACIÓN DE API (GEMINI) - MODELOS DISPONIBLES
# =============================================================================

def get_gemini_client(api_key):
    """Inicializa cliente de Gemini con modelo disponible"""
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        # ✅ MODELOS DISPONIBLES (elige uno):
        # 'gemini-1.5-flash' → Rápido y gratuito (RECOMENDADO)
        # 'gemini-1.5-pro' → Más potente pero con límites
        # 'gemini-pro' → Versión estable clásica
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configurando Gemini: {e}")
        return None

def analizar_con_gemini(model, top_combinaciones, config, historial_stats):
    """Envía las top combinaciones a Gemini para análisis cualitativo."""
    if not model:
        return None
    
    prompt = f"""
Eres un experto en estadística aplicada, teoría de valores extremos y análisis de sistemas estocásticos.

## CONTEXTO DEL SISTEMA PIV-60
- Universo: 46 números (0-45)
- Sorteo: 6 números
- Protocolo: PIV-60 (Ingeniería Probabilística)
- Combinaciones analizadas: {len(top_combinaciones)}

## PARÁMETROS CALIBRADOS
- Gumbel μ={config['gumbel_mu']:.2f}, β={config['gumbel_beta']:.2f}
- Gauss μ={config['gauss_mean']:.1f}, σ={config['gauss_std']:.1f}
- Pesos IPC: Hist={config['omega_hist']}, Rec={config['omega_rec']}, Gumbel={config['omega_gum']}

## ESTADÍSTICAS HISTÓRICAS
{historial_stats}

## TOP 5 COMBINACIONES CON MÉTRICAS
{format_top_combos_for_llm(top_combinaciones[:5])}

## TAREA
Analiza estas combinaciones y responde:

1. **¿Cuál combinación tiene el perfil estadístico más sólido?** (considerando balance entre IPC, zona energética, y distribución de atrasos)

2. **¿Qué patrones observas en las combinaciones mejor rankeadas?** (ej: predominio de números fríos/calientes, suma concentrada, etc.)

3. **¿Hay alguna anomalía o riesgo en estas predicciones?** (ej: sobre-dependencia de un número, suma muy extrema, etc.)

4. **Recomendación final:** Si tuvieras que seleccionar UNA combinación para jugar, ¿cuál sería y por qué?

Sé honesto sobre las limitaciones: ningún modelo puede predecir sorteos aleatorios con certeza. Esto es análisis estadístico, no predicción mágica.

Responde en español, de forma clara y estructurada.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en análisis Gemini: {str(e)}"

def format_top_combos_for_llm(combos):
    """Formatea combinaciones para el prompt del LLM"""
    text = ""
    for i, c in enumerate(combos, 1):
        text += f"""
#{i}: {c['Combinación']}
   - Score: {c['Score']:.4f} | IPC: {c['IPC']:.4f}
   - Zona: {c['Zona']} | S: {c['S']:.1f}
   - Suma: {c['Suma']}
   - Frec Hist: {c['F_Hist']:.3f} | Reciente: {c['V_60']:.3f} | Gumbel: {c['T_Gumbel']:.3f}
"""
    return text

# =============================================================================
# CLASE PIV-60
# =============================================================================

class CalibradorDinamicoPIV60:
    def __init__(self, df_historial, df_datos):
        self.historial = self._procesar_historial(df_historial)
        self.datos_actuales = self._procesar_datos(df_datos)
        self.config = {}
        
    def _procesar_historial(self, df):
        cols = df.columns.tolist()
        unique_cols = []
        seen = {}
        for col in cols:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        df.columns = unique_cols
        
        historial = []
        num_cols = [c for c in df.columns if c.startswith('C') and c != 'Fecha']
        
        for _, row in df.iterrows():
            nums = set()
            for c in num_cols:
                try:
                    n = int(row[c])
                    if 0 <= n <= 45:
                        nums.add(n)
                except:
                    continue
            if len(nums) >= 5:
                historial.append(sorted(nums))
        
        return historial

    def _procesar_datos(self, df):
        df.columns = df.columns.str.strip().str.lower()
        datos = {}
        for _, row in df.iterrows():
            datos[int(row['numero'])] = {
                'atraso': int(row['atraso']),
                'frecuencia': int(row['frecuencia'])
            }
        return datos

    def calcular_parametros_gumbel(self):
        atrasos = [info['atraso'] for info in self.datos_actuales.values()]
        loc, scale = gumbel_r.fit(atrasos, floc=0)
        return loc, scale

    def calcular_parametros_gauss(self):
        sumas = [sum(sorteo) for sorteo in self.historial]
        return np.mean(sumas), np.std(sumas)

    def ejecutar_calibracion(self):
        mu, beta = self.calcular_parametros_gumbel()
        g_mean, g_std = self.calcular_parametros_gauss()
        
        Cs_hist = []
        for i in range(60, len(self.historial)):
            estado = self._reconstruir_estado(i)
            C = sum(info['atraso'] for info in estado.values()) + 40
            Cs_hist.append(C)
        
        if Cs_hist:
            p25, p50, p75 = np.percentile(Cs_hist, [25, 50, 75])
        else:
            p25, p50, p75 = 280, 320, 360
        
        self.config = {
            'gumbel_mu': mu, 'gumbel_beta': beta,
            'gauss_mean': g_mean, 'gauss_std': max(g_std, 20),
            'omega_hist': 0.25, 'omega_rec': 0.40, 'omega_gum': 0.35,
            'zona_inercia': p75, 'zona_equilibrio': (p25, p75), 'zona_ruptura': p25,
            'constante_k': 40
        }
        return self.config

    def _reconstruir_estado(self, hasta_indice):
        atrasos = {i: 0 for i in range(46)}
        frecuencias = {i: 0 for i in range(46)}
        for s in self.historial[:hasta_indice]:
            for n in s:
                frecuencias[n] += 1
                atrasos[n] = 0
            for n in range(46):
                if n not in s:
                    atrasos[n] += 1
        return {n: {'atraso': atrasos[n], 'frecuencia': frecuencias[n]} for n in range(46)}

def calcular_IPC_calibrado(combinacion, datos, config):
    nums = list(combinacion)
    freqs = [datos[str(n)]['frecuencia'] for n in nums if str(n) in datos]
    max_freq = max(info['frecuencia'] for info in datos.values())
    F_hist = (np.mean(freqs) / max_freq) if freqs and max_freq > 0 else 0
    
    scores_rec = [np.exp(-datos[str(n)]['atraso'] / 5.0) for n in nums if str(n) in datos]
    V_60 = np.mean(scores_rec) if scores_rec else 0
    
    tensions = []
    for n in nums:
        if str(n) in datos:
            at = datos[str(n)]['atraso']
            prob = gumbel_r.pdf(at, loc=config['gumbel_mu'], scale=config['gumbel_beta'])
            tensions.append(prob * at)
    T_g = np.sum(tensions)
    
    suma = sum(nums)
    diff = abs(suma - config['gauss_mean'])
    phi_gauss = 0.5 * (diff / config['gauss_std']) ** 2 if config['gauss_std'] > 0 else 0
    
    ipc = config['omega_hist']*F_hist + config['omega_rec']*V_60 + config['omega_gum']*T_g - phi_gauss
    
    return {'IPC_Total': ipc, 'F_hist': F_hist, 'V_60': V_60, 'T_g': T_g, 
            'phi_gauss': phi_gauss, 'suma': suma}

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.title("🤖 PIV-60 v5.0 + Gemini IA")
st.markdown("**Documento:** PIP-2026-X46 | **IA:** Google Gemini 1.5 Flash")

with st.sidebar:
    st.header("📁 Archivos")
    archivo_historial = st.file_uploader("Historial_Tradicional.csv", type=['csv'], key='hist')
    archivo_datos = st.file_uploader("datos_actuales.csv", type=['csv'], key='datos')
    
    st.markdown("---")
    st.header("🤖 Configuración Gemini")
    usar_llm = st.checkbox("Activar análisis con Gemini", value=False)
    
    api_key_input = st.text_input("Gemini API Key", type="password", 
                                   help="Obtén tu key en https://aistudio.google.com/app/apikey",
                                   value="")
    
    # Selector de modelo
    modelo_seleccionado = st.selectbox(
        "Modelo Gemini",
        options=['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro'],
        index=0,
        help="Flash = rápido/gratuito, Pro = más potente"
    )
    
    st.markdown("---")
    top_n = st.slider("Top combinaciones", 5, 50, 15)
    ejecutar = st.button("🚀 Ejecutar", type="primary", use_container_width=True)

if ejecutar:
    if archivo_historial is None or archivo_datos is None:
        st.error("❌ Sube ambos archivos CSV")
        st.stop()
    
    try:
        with st.spinner("📥 Cargando datos..."):
            df_hist = pd.read_csv(archivo_historial)
            df_datos = pd.read_csv(archivo_datos)
            
        with st.spinner("⚙️ Auto-calibrando..."):
            calibrador = CalibradorDinamicoPIV60(df_hist, df_datos)
            config = calibrador.ejecutar_calibracion()
            
        st.success("✅ Calibración completada")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gumbel μ", f"{config['gumbel_mu']:.2f}")
            st.metric("Gumbel β", f"{config['gumbel_beta']:.2f}")
        with col2:
            st.metric("Gauss μ", f"{config['gauss_mean']:.1f}")
            st.metric("Gauss σ", f"{config['gauss_std']:.1f}")
        with col3:
            st.metric("ω Histórico", f"{config['omega_hist']:.3f}")
            st.metric("ω Reciente", f"{config['omega_rec']:.3f}")
            st.metric("ω Gumbel", f"{config['omega_gum']:.3f}")
        
        # Generar combinaciones
        with st.spinner("🔄 Generando combinaciones..."):
            datos = calibrador.datos_actuales
            historial = calibrador.historial
            
            st.write(f"🔍 Historial cargado: {len(historial)} sorteos")
            if len(historial) > 0:
                st.write(f"   • Primer sorteo: {historial[0]}")
                st.write(f"   • Último sorteo: {historial[-1]}")
            
            momento = [n for n, info in datos.items() if info['atraso'] == 0]
            masa = [n for n, info in datos.items() if 1 <= info['atraso'] <= 9]
            tension = [n for n, info in datos.items() if info['atraso'] >= 15]
            
            suma_atrasos = sum(info['atraso'] for info in datos.values())
            C_actual = suma_atrasos + config['constante_k']
            
            st.info(f"📊 Clasificación: **Momento**={len(momento)}, **Masa**={len(masa)}, **Tensión**={len(tension)}")
            st.write(f"   • Momento (A=0): {sorted(momento)}")
            st.write(f"   • Tensión (A≥15): {sorted(tension)}")
            
            combinaciones = []
            for m in momento:
                for mc in combinations(masa, 4):
                    for t in tension:
                        combo = tuple(sorted([m] + list(mc) + [t]))
                        if len(set(combo)) == 6:
                            combinaciones.append(combo)
            
            st.write(f"✅ {len(combinaciones):,} combinaciones generadas")
            
            resultados = []
            for combo in combinaciones:
                suma_combo = sum(combo)
                if not (100 <= suma_combo <= 170):
                    continue
                    
                atrasos_c = [datos[n]['atraso'] for n in combo]
                S = C_actual - sum(atrasos_c)
                
                if S > config['zona_inercia']:
                    zona, peso_z = 'INERCIA', 1.0
                elif config['zona_equilibrio'][0] < S < config['zona_equilibrio'][1]:
                    zona, peso_z = 'EQUILIBRIO', 1.5
                elif S < config['zona_ruptura']:
                    zona, peso_z = 'RUPTURA', 1.2
                else:
                    zona, peso_z = 'TRANSICIÓN', 0.8
                
                ipc_d = calcular_IPC_calibrado(combo, datos, config)
                score = ipc_d['IPC_Total'] * peso_z
                
                resultados.append({
                    'Rank': 0, 'Combinación': ' - '.join(map(str, combo)),
                    'N1': combo[0], 'N2': combo[1], 'N3': combo[2], 
                    'N4': combo[3], 'N5': combo[4], 'N6': combo[5],
                    'Score': score, 'IPC': ipc_d['IPC_Total'],
                    'S': S, 'Zona': zona, 'Suma': suma_combo,
                    'F_Hist': ipc_d['F_hist'], 'V_60': ipc_d['V_60'],
                    'T_Gumbel': ipc_d['T_g'], 'Penal_Gauss': ipc_d['phi_gauss']
                })
            
            if not resultados:
                st.error("❌ No se generaron combinaciones válidas.")
                st.stop()
            
            resultados.sort(key=lambda x: x['Score'], reverse=True)
            for i, r in enumerate(resultados, 1):
                r['Rank'] = i
            
            finalistas = resultados[:top_n]
            
        # ==================== MOSTRAR RESULTADOS ====================
        
        st.subheader(f"🏆 Top {top_n} Combinaciones")
        for i, item in enumerate(finalistas, 1):
            with st.expander(f"#{i:02d} | {item['Combinación']} | Score: {item['Score']:.4f}", expanded=(i<=5)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Zona:** {item['Zona']}")
                    st.write(f"**S:** {item['S']:.1f}")
                    st.write(f"**Suma:** {item['Suma']}")
                with col_b:
                    st.write(f"**IPC:** {item['IPC']:.4f}")
                    st.write(f"**Hist:** {item['F_Hist']:.3f}")
                    st.write(f"**Rec:** {item['V_60']:.3f}")
                    st.write(f"**Gumbel:** {item['T_Gumbel']:.3f}")
        
        # ==================== ANÁLISIS CON GEMINI ====================
        
        if usar_llm:
            st.subheader("🤖 Análisis Inteligente con Gemini")
            
            if not api_key_input:
                st.warning("⚠️ Ingresa tu API Key de Gemini para activar el análisis con IA.")
            else:
                with st.spinner("🧠 Consultando a Gemini..."):
                    # Configurar con el modelo seleccionado
                    genai.configure(api_key=api_key_input)
                    model = genai.GenerativeModel(modelo_seleccionado)
                    
                    if not model:
                        st.error("❌ Error al configurar Gemini. Verifica tu API Key.")
                    else:
                        if historial and len(historial) > 0:
                            sumas_hist = [sum(s) for s in historial]
                            hist_stats = f"""
                            - Total sorteos en historial: {len(historial)}
                            - Suma promedio histórica: {np.mean(sumas_hist):.1f}
                            - Suma máxima histórica: {max(sumas_hist)}
                            - Suma mínima histórica: {min(sumas_hist)}
                            - Desviación estándar: {np.std(sumas_hist):.1f}
                            """
                        else:
                            hist_stats = """
                            - Historial: No disponible
                            - Usando configuración por defecto
                            """
                        
                        analisis_gemini = analizar_con_gemini(model, finalistas, config, hist_stats)
                        
                        if analisis_gemini and not analisis_gemini.startswith("Error"):
                            st.markdown("### 📝 Informe de Gemini")
                            st.markdown(analisis_gemini)
                            
                            st.download_button(
                                label="📥 Descargar Informe Gemini (Markdown)",
                                data=analisis_gemini,
                                file_name="informe_gemini_piv60.md",
                                mime="text/markdown"
                            )
                            
                            st.success("✅ Análisis completado con Gemini")
                        else:
                            st.error(analisis_gemini)
        
        # ==================== TABLA Y DESCARGA ====================
        
        st.subheader(f"📋 Todas las Combinaciones ({len(resultados):,})")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados[['Rank', 'Combinación', 'Score', 'IPC', 'Zona', 'Suma', 'S']], 
                    use_container_width=True, height=600)
        
        csv = df_resultados.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(label=f"📥 Descargar CSV ({len(resultados):,} combinaciones)",
                          data=csv, file_name=f"PIV60_Resultados_{len(resultados)}.csv",
                          mime="text/csv")
        
        # ==================== MAPA DE CALOR ====================
        
        st.subheader("🎨 Mapa de Calor")
        numeros = sorted(list(datos.keys()))
        matriz = np.full((6, 8), np.nan)
        for i, n in enumerate(numeros):
            matriz[i//8, i%8] = datos[n]['atraso']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(matriz, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   linewidths=1, linecolor='gray', ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        for i, n in enumerate(numeros):
            r, c = i//8, i%8
            ax.text(c+0.5, r+0.8, str(n), ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=10)
        st.pyplot(fig)
        
        st.success("✅ Análisis completado")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
else:
    st.info("👈 Sube los archivos y presiona Ejecutar")

st.markdown("---")
st.caption("⚠️ **Advertencia:** Este sistema es para investigación estadística. Ningún modelo puede garantizar aciertos en sorteos aleatorios.")
