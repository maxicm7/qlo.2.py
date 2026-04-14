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
from datetime import datetime

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

st.set_page_config(page_title="PIV-60 v6.0 + Backtesting", page_icon="🔬", layout="wide")

# =============================================================================
# CONFIGURACIÓN DE API (GEMINI)
# =============================================================================

def get_gemini_client(api_key):
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configurando Gemini: {e}")
        return None

def analizar_con_gemini(model, top_combinaciones, config, historial_stats):
    if not model:
        return None
    
    prompt = f"""
Eres un experto en estadística aplicada y análisis de sistemas estocásticos.

## CONTEXTO PIV-60
- Universo: 46 números (0-45)
- Protocolo: Ingeniería Probabilística
- Combinaciones analizadas: {len(top_combinaciones)}

## PARÁMETROS CALIBRADOS
- Gumbel μ={config['gumbel_mu']:.2f}, β={config['gumbel_beta']:.2f}
- Gauss μ={config['gauss_mean']:.1f}, σ={config['gauss_std']:.1f}
- Pesos IPC: Hist={config['omega_hist']}, Rec={config['omega_rec']}, Gumbel={config['omega_gum']}

## ESTADÍSTICAS HISTÓRICAS
{historial_stats}

## TOP 5 COMBINACIONES
{format_top_combos_for_llm(top_combinaciones[:5])}

## TAREA
1. ¿Cuál combinación tiene el perfil estadístico más sólido?
2. ¿Qué patrones observas en las combinaciones mejor rankeadas?
3. ¿Hay alguna anomalía o riesgo?
4. Recomendación final: ¿UNA combinación para jugar y por qué?

Sé honesto sobre limitaciones. Responde en español, claro y estructurado.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error en análisis Gemini: {str(e)}"

def format_top_combos_for_llm(combos):
    text = ""
    for i, c in enumerate(combos, 1):
        text += f"""
#{i}: {c['Combinación']}
   - Score: {c['Score']:.4f} | IPC: {c['IPC']:.4f}
   - Zona: {c['Zona']} | S: {c['S']:.1f}
   - Suma: {c['Suma']}
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
        fechas = []
        num_cols = [c for c in df.columns if c.startswith('C') and c != 'Fecha']
        
        for idx, row in df.iterrows():
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
                # Intentar extraer fecha si existe
                if 'Fecha' in row:
                    fechas.append(row['Fecha'])
                else:
                    fechas.append(f"Sorteo {idx+1}")
        
        return historial, fechas

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
        sumas = [sum(sorteo) for sorteo in self.historial[0]]
        return np.mean(sumas), np.std(sumas)

    def ejecutar_calibracion(self):
        mu, beta = self.calcular_parametros_gumbel()
        g_mean, g_std = self.calcular_parametros_gauss()
        
        Cs_hist = []
        for i in range(60, len(self.historial[0])):
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
        for s in self.historial[0][:hasta_indice]:
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
# MÓDULO DE BACKTESTING WALK-FORWARD
# =============================================================================

def ejecutar_backtesting(historial, fechas, config, ventana_min=20, top_evaluar=10):
    """
    Ejecuta backtesting walk-forward sobre el historial completo.
    Para cada sorteo, recalcula atrasos hasta esa fecha y genera predicciones.
    """
    resultados_bt = []
    total_sorteos = len(historial)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(ventana_min, total_sorteos):
        # Actualizar progreso
        progreso = (i - ventana_min) / (total_sorteos - ventana_min)
        progress_bar.progress(progreso)
        status_text.text(f"Evaluando sorteo {i+1}/{total_sorteos} ({fechas[i]})")
        
        # 1. Reconstruir estado HASTA el sorteo anterior (walk-forward)
        atrasos = {n: 0 for n in range(46)}
        frecuencias = {n: 0 for n in range(46)}
        
        for s in historial[:i]:
            for n in s:
                frecuencias[n] += 1
                atrasos[n] = 0
            for n in range(46):
                if n not in s:
                    atrasos[n] += 1
        
        datos_historicos = {str(n): {'atraso': atrasos[n], 'frecuencia': frecuencias[n]} for n in range(46)}
        
        # 2. Calcular C actual hasta esa fecha
        suma_atrasos = sum(atrasos.values())
        C_actual = suma_atrasos + 40
        
        # 3. Clasificar números según estado histórico
        momento = [n for n in range(46) if atrasos[n] == 0]
        masa = [n for n in range(46) if 1 <= atrasos[n] <= 9]
        tension = [n for n in range(46) if atrasos[n] >= 15]
        
        # 4. Generar combinaciones (limitar para velocidad)
        combinaciones = []
        if len(momento) >= 1 and len(masa) >= 4 and len(tension) >= 1:
            for m in momento[:10]:  # Limitar para velocidad
                for mc in combinations(masa[:20], 4):  # Limitar para velocidad
                    for t in tension[:5]:  # Limitar para velocidad
                        combo = tuple(sorted([m] + list(mc) + [t]))
                        if len(set(combo)) == 6 and 100 <= sum(combo) <= 170:
                            combinaciones.append(combo)
                        if len(combinaciones) >= 5000:  # Máximo 5k por sorteo
                            break
                    if len(combinaciones) >= 5000:
                        break
                if len(combinaciones) >= 5000:
                    break
        
        # 5. Rankear combinaciones
        scored = []
        for combo in combinaciones:
            atrasos_c = [atrasos[n] for n in combo]
            S = C_actual - sum(atrasos_c)
            
            if S > config['zona_inercia']:
                zona, peso_z = 'INERCIA', 1.0
            elif config['zona_equilibrio'][0] < S < config['zona_equilibrio'][1]:
                zona, peso_z = 'EQUILIBRIO', 1.5
            elif S < config['zona_ruptura']:
                zona, peso_z = 'RUPTURA', 1.2
            else:
                zona, peso_z = 'TRANSICIÓN', 0.8
            
            ipc_d = calcular_IPC_calibrado(combo, datos_historicos, config)
            score = ipc_d['IPC_Total'] * peso_z
            scored.append({'combo': combo, 'score': score})
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_combos = [c['combo'] for c in scored[:top_evaluar]]
        
        # 6. Comparar con resultado real
        resultado_real = set(historial[i])
        mejores_aciertos = 0
        mejor_combo = None
        
        for combo in top_combos:
            aciertos = len(set(combo) & resultado_real)
            if aciertos > mejores_aciertos:
                mejores_aciertos = aciertos
                mejor_combo = combo
        
        resultados_bt.append({
            'indice': i,
            'fecha': fechas[i],
            'aciertos': mejores_aciertos,
            'resultado_real': sorted(resultado_real),
            'prediccion_top': list(mejor_combo) if mejor_combo else [],
            'total_combos_evaluadas': len(scored)
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Backtesting completado")
    
    return pd.DataFrame(resultados_bt)

def calcular_metricas_backtesting(df_bt):
    """Calcula métricas agregadas de rendimiento."""
    if len(df_bt) == 0:
        return {}
    
    total = len(df_bt)
    return {
        'total_sorteos': total,
        'hit_rate_3plus': (df_bt['aciertos'] >= 3).sum() / total,
        'hit_rate_4plus': (df_bt['aciertos'] >= 4).sum() / total,
        'hit_rate_5plus': (df_bt['aciertos'] >= 5).sum() / total,
        'hit_rate_6': (df_bt['aciertos'] == 6).sum() / total,
        'aciertos_promedio': df_bt['aciertos'].mean(),
        'desviacion': df_bt['aciertos'].std(),
        'mejor_racha_4plus': calcular_mejor_racha(df_bt['aciertos'] >= 4),
        'mejor_racha_5plus': calcular_mejor_racha(df_bt['aciertos'] >= 5)
    }

def calcular_mejor_racha(serie_bool):
    max_racha = current = 0
    for val in serie_bool:
        if val:
            current += 1
            max_racha = max(max_racha, current)
        else:
            current = 0
    return max_racha

def calcular_probabilidades_azar_6_46():
    """Probabilidades teóricas para lotería 6/46."""
    from math import comb
    total = comb(46, 6)
    return {
        '3/6': comb(6, 3) * comb(40, 3) / total,
        '4/6': comb(6, 4) * comb(40, 2) / total,
        '5/6': comb(6, 5) * comb(40, 1) / total,
        '6/6': comb(6, 6) * comb(40, 0) / total
    }

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.title("🔬 PIV-60 v6.0 + Backtesting")
st.markdown("**Documento:** PIP-2026-X46 | **Validación:** Walk-Forward Testing")

# Pestañas
tab_prediccion, tab_backtesting = st.tabs(["🔮 Predicción Actual", "📊 Backtesting Histórico"])

# =============================================================================
# PESTAÑA 1: PREDICCIÓN ACTUAL
# =============================================================================

with tab_prediccion:
    st.header("Predicción para Próximo Sorteo")
    
    with st.sidebar:
        st.header("📁 Archivos")
        archivo_historial = st.file_uploader("Historial_Tradicional.csv", type=['csv'], key='hist')
        archivo_datos = st.file_uploader("datos_actuales.csv", type=['csv'], key='datos')
        
        st.markdown("---")
        st.header("🤖 Configuración Gemini")
        usar_llm = st.checkbox("Activar análisis con Gemini", value=False)
        api_key_input = st.text_input("Gemini API Key", type="password", value="")
        
        st.markdown("---")
        top_n = st.slider("Top combinaciones", 5, 50, 15)
        ejecutar_pred = st.button("🚀 Ejecutar Predicción", type="primary", use_container_width=True)
    
    if ejecutar_pred:
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
            
            with st.spinner("🔄 Generando combinaciones..."):
                datos = calibrador.datos_actuales
                historial = calibrador.historial[0]
                
                momento = [n for n, info in datos.items() if info['atraso'] == 0]
                masa = [n for n, info in datos.items() if 1 <= info['atraso'] <= 9]
                tension = [n for n, info in datos.items() if info['atraso'] >= 15]
                
                suma_atrasos = sum(info['atraso'] for info in datos.values())
                C_actual = suma_atrasos + config['constante_k']
                
                st.info(f"📊 Clasificación: **Momento**={len(momento)}, **Masa**={len(masa)}, **Tensión**={len(tension)}")
                
                combinaciones = []
                for m in momento:
                    for mc in combinations(masa, 4):
                        for t in tension:
                            combo = tuple(sorted([m] + list(mc) + [t]))
                            if len(set(combo)) == 6 and 100 <= sum(combo) <= 170:
                                combinaciones.append(combo)
                
                st.write(f"✅ {len(combinaciones):,} combinaciones generadas")
                
                resultados = []
                for combo in combinaciones:
                    atrasos_c = [datos[str(n)]['atraso'] for n in combo]
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
                        'S': S, 'Zona': zona, 'Suma': sum(combo),
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
            
            # Análisis con Gemini
            if usar_llm and api_key_input:
                st.subheader("🤖 Análisis con Gemini")
                with st.spinner("🧠 Consultando a Gemini..."):
                    model = get_gemini_client(api_key_input)
                    if model:
                        sumas_hist = [sum(s) for s in historial]
                        hist_stats = f"""
                        - Total sorteos: {len(historial)}
                        - Suma promedio: {np.mean(sumas_hist):.1f}
                        - Suma máx: {max(sumas_hist)}, mín: {min(sumas_hist)}
                        """
                        analisis = analizar_con_gemini(model, finalistas, config, hist_stats)
                        if analisis:
                            st.markdown(analisis)
            
            # Tabla y descarga
            st.subheader(f"📋 Todas las Combinaciones ({len(resultados):,})")
            df_resultados = pd.DataFrame(resultados)
            st.dataframe(df_resultados[['Rank', 'Combinación', 'Score', 'IPC', 'Zona', 'Suma', 'S']], 
                        use_container_width=True, height=600)
            
            csv = df_resultados.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label=f"📥 Descargar CSV", data=csv, 
                              file_name=f"PIV60_Prediccion_{len(resultados)}.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# =============================================================================
# PESTAÑA 2: BACKTESTING
# =============================================================================

with tab_backtesting:
    st.header("🔬 Validación Histórica Walk-Forward")
    st.info("ℹ️ El backtesting recalcula los atrasos para CADA fecha histórica, simulando cómo habría funcionado el modelo en el pasado.")
    
    if archivo_historial is None or archivo_datos is None:
        st.warning("⚠️ Primero sube los archivos en la pestaña 'Predicción Actual'")
    else:
        with st.expander("⚙️ Parámetros de Backtesting", expanded=True):
            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                ventana_min_bt = st.number_input("Sorteos mínimos para iniciar", 10, 50, 20)
                top_evaluar_bt = st.number_input("Top predicciones a evaluar por sorteo", 1, 50, 10)
            with col_bt2:
                ejecutar_bt = st.button("🔄 Ejecutar Backtesting", type="primary", use_container_width=True)
        
        if ejecutar_bt:
            try:
                with st.spinner("📥 Cargando datos..."):
                    df_hist = pd.read_csv(archivo_historial)
                    df_datos = pd.read_csv(archivo_datos)
                
                with st.spinner("⚙️ Calibrando modelo..."):
                    calibrador = CalibradorDinamicoPIV60(df_hist, df_datos)
                    config = calibrador.ejecutar_calibracion()
                
                historial = calibrador.historial[0]
                fechas = calibrador.historial[1]
                
                st.info(f"📊 Historial cargado: {len(historial)} sorteos")
                
                # Ejecutar backtesting
                df_bt = ejecutar_backtesting(historial, fechas, config, ventana_min_bt, top_evaluar_bt)
                metricas = calcular_metricas_backtesting(df_bt)
                prob_azar = calcular_probabilidades_azar_6_46()
                
                # Métricas principales
                st.subheader("📊 Métricas de Rendimiento")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Sorteos evaluados", metricas.get('total_sorteos', 0))
                with m2:
                    st.metric("Hit-rate ≥3", f"{metricas.get('hit_rate_3plus', 0)*100:.1f}%")
                with m3:
                    st.metric("Hit-rate ≥4", f"{metricas.get('hit_rate_4plus', 0)*100:.2f}%")
                with m4:
                    st.metric("Hit-rate ≥5", f"{metricas.get('hit_rate_5plus', 0)*100:.3f}%")
                with m5:
                    st.metric("Aciertos promedio", f"{metricas.get('aciertos_promedio', 0):.2f}")
                
                # Comparación con azar
                st.subheader("🎲 Comparación con Línea Base Aleatoria (6/46)")
                col_az1, col_az2 = st.columns(2)
                with col_az1:
                    st.markdown("** Modelo PIV-60 (observado):**")
                    st.write(f"- ≥3 aciertos: {metricas.get('hit_rate_3plus', 0)*100:.2f}%")
                    st.write(f"- ≥4 aciertos: {metricas.get('hit_rate_4plus', 0)*100:.4f}%")
                    st.write(f"- ≥5 aciertos: {metricas.get('hit_rate_5plus', 0)*100:.5f}%")
                
                with col_az2:
                    st.markdown("**🎲 Azar puro (teórico):**")
                    st.write(f"- ≥3 aciertos: {prob_azar['3/6']*100:.3f}%")
                    st.write(f"- ≥4 aciertos: {prob_azar['4/6']*100:.4f}%")
                    st.write(f"- ≥5 aciertos: {prob_azar['5/6']*100:.5f}%")
                
                # Ratio de mejora
                if metricas.get('hit_rate_4plus', 0) > 0:
                    ratio_4 = metricas['hit_rate_4plus'] / prob_azar['4/6']
                    ratio_5 = metricas['hit_rate_5plus'] / prob_azar['5/6'] if prob_azar['5/6'] > 0 else 0
                    st.info(f"📈 El modelo es **{ratio_4:.1f}x** mejor que el azar para ≥4 aciertos")
                    if ratio_5 > 1:
                        st.success(f"🚀 El modelo es **{ratio_5:.0f}x** mejor que el azar para ≥5 aciertos")
                
                # Gráficos
                st.subheader("📈 Visualizaciones")
                
                # Evolución temporal
                fig1, ax1 = plt.subplots(figsize=(12, 5))
                ax1.plot(df_bt['indice'], df_bt['aciertos'], marker='o', linestyle='-', linewidth=1, markersize=4)
                ax1.axhline(y=metricas.get('aciertos_promedio', 0), color='r', linestyle='--', label=f"Promedio ({metricas.get('aciertos_promedio', 0):.2f})")
                ax1.axhline(y=0.78, color='gray', linestyle=':', label="Azar esperado (0.78)")
                ax1.set_xlabel("Sorteo")
                ax1.set_ylabel("Aciertos")
                ax1.set_title("Evolución Temporal de Aciertos")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # Distribución de aciertos
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                dist = df_bt['aciertos'].value_counts().sort_index()
                ax2.bar(dist.index, dist.values, color='steelblue', edgecolor='black')
                ax2.set_xlabel("Aciertos")
                ax2.set_ylabel("Frecuencia")
                ax2.set_title("Distribución de Aciertos")
                ax2.set_xticks(range(7))
                ax2.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig2)
                
                # Tabla de resultados
                st.subheader("📋 Resultados Detallados por Sorteo")
                st.dataframe(df_bt[['fecha', 'aciertos', 'resultado_real', 'prediccion_top']], 
                            use_container_width=True, height=400)
                
                # Descarga
                csv_bt = df_bt.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(label="📥 Descargar Resultados Backtesting (CSV)",
                                  data=csv_bt, file_name="backtesting_piv60.csv", mime="text/csv")
                
                # Informe resumen
                informe = f"""# Informe de Backtesting PIV-60
**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Sorteos evaluados:** {metricas.get('total_sorteos', 0)}
**Ventana mínima:** {ventana_min_bt}

## Métricas Clave
- Hit-rate ≥4 aciertos: {metricas.get('hit_rate_4plus', 0)*100:.3f}%
- Hit-rate ≥5 aciertos: {metricas.get('hit_rate_5plus', 0)*100:.5f}%
- Aciertos promedio: {metricas.get('aciertos_promedio', 0):.2f} ± {metricas.get('desviacion', 0):.2f}

## Comparación vs. Azar
- Mejora para ≥4 aciertos: {ratio_4:.1f}x
- Mejora para ≥5 aciertos: {ratio_5:.0f}x

## Conclusión
{'✅ El modelo muestra rendimiento significativamente superior al azar.' if ratio_4 > 5 else '⚠️ El rendimiento requiere más datos para validación.'}
"""
                st.download_button(label="📥 Descargar Informe (Markdown)",
                                  data=informe, file_name="informe_backtesting.md", mime="text/markdown")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

st.markdown("---")
st.caption("⚠️ **Advertencia:** Este sistema es para investigación estadística. Ningún modelo puede garantizar aciertos en sorteos aleatorios.")
