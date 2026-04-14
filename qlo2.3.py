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

st.set_page_config(page_title="PIV-60 v6.5 + Métricas Avanzadas", page_icon="🔬", layout="wide")

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
2. ¿Qué patrones observas?
3. ¿Hay alguna anomalía o riesgo?
4. Recomendación final: ¿UNA combinación y por qué?

Sé honesto sobre limitaciones. Responde en español.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def format_top_combos_for_llm(combos):
    text = ""
    for i, c in enumerate(combos, 1):
        text += f"#{i}: {c['Combinación']} | Score: {c['Score']:.4f}\n"
    return text

# =============================================================================
# CLASE PIV-60
# =============================================================================

class CalibradorDinamicoPIV60:
    def __init__(self, df_historial, df_datos):
        self.historial, self.fechas = self._procesar_historial(df_historial)
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
        num_cols = [c for c in df.columns if c.startswith('C')]
        
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
                if 'Fecha' in row:
                    fechas.append(str(row['Fecha']))
                else:
                    fechas.append(f"Sorteo {idx+1}")
        
        return historial, fechas

    def _procesar_datos(self, df):
        df.columns = df.columns.str.strip().str.lower()
        datos = {}
        for _, row in df.iterrows():
            # 🔧 CLAVES COMO ENTEROS (consistente)
            num = int(row['numero'])
            datos[num] = {
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

# 🔧 FUNCIÓN CORREGIDA: Acceder con claves enteras
def calcular_IPC_calibrado(combinacion, datos, config):
    nums = list(combinacion)
    # 🔧 Usar claves enteras, no strings
    freqs = [datos[n]['frecuencia'] for n in nums if n in datos]
    max_freq = max(info['frecuencia'] for info in datos.values())
    F_hist = (np.mean(freqs) / max_freq) if freqs and max_freq > 0 else 0
    
    # 🔧 Usar claves enteras
    scores_rec = [np.exp(-datos[n]['atraso'] / 5.0) for n in nums if n in datos]
    V_60 = np.mean(scores_rec) if scores_rec else 0
    
    tensions = []
    for n in nums:
        if n in datos:
            at = datos[n]['atraso']
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
# MÉTRICAS AVANZADAS
# =============================================================================

def calcular_sharpe_ratio(aciertos, costo_jugada=1, premio_3=100, premio_4=5000, premio_5=50000, premio_6=1000000):
    retornos = []
    for ac in aciertos:
        if ac >= 6:
            retorno = (premio_6 - costo_jugada) / costo_jugada
        elif ac >= 5:
            retorno = (premio_5 - costo_jugada) / costo_jugada
        elif ac >= 4:
            retorno = (premio_4 - costo_jugada) / costo_jugada
        elif ac >= 3:
            retorno = (premio_3 - costo_jugada) / costo_jugada
        else:
            retorno = -1
        retornos.append(retorno)
    
    if len(retornos) < 2:
        return 0, 0, 0
    
    retorno_promedio = np.mean(retornos)
    volatilidad = np.std(retornos)
    sharpe = retorno_promedio / volatilidad if volatilidad > 0 else 0
    
    return sharpe, retorno_promedio, volatilidad

def calcular_valor_esperado(hit_rates, premios, costo_jugada=1):
    ev = 0
    for aciertos, (prob, premio) in hit_rates.items():
        ev += prob * premio
    ev -= costo_jugada
    return ev

def calcular_confianza(score, score_max, score_min):
    if score_max == score_min:
        return 50
    return int(((score - score_min) / (score_max - score_min)) * 100)

def calcular_probabilidades_azar_6_46():
    from math import comb
    total = comb(46, 6)
    return {
        '3/6': comb(6, 3) * comb(40, 3) / total,
        '4/6': comb(6, 4) * comb(40, 2) / total,
        '5/6': comb(6, 5) * comb(40, 1) / total,
        '6/6': comb(6, 6) * comb(40, 0) / total
    }

# =============================================================================
# BACKTESTING RÁPIDO
# =============================================================================

def ejecutar_backtesting_rapido(historial, fechas, config, ventana_min=20, top_evaluar=10, max_sorteos=20):
    resultados_bt = []
    total_sorteos = len(historial)
    inicio = max(ventana_min, total_sorteos - max_sorteos)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(inicio, total_sorteos):
        progreso = (i - inicio) / (total_sorteos - inicio)
        progress_bar.progress(progreso)
        status_text.text(f"Evaluando {fechas[i]} ({i+1}/{total_sorteos})")
        
        atrasos = {n: 0 for n in range(46)}
        frecuencias = {n: 0 for n in range(46)}
        
        for s in historial[:i]:
            for n in s:
                frecuencias[n] += 1
                atrasos[n] = 0
            for n in range(46):
                if n not in s:
                    atrasos[n] += 1
        
        # 🔧 Claves enteras
        datos_historicos = {n: {'atraso': atrasos[n], 'frecuencia': frecuencias[n]} for n in range(46)}
        suma_atrasos = sum(atrasos.values())
        C_actual = suma_atrasos + 40
        
        momento = [n for n in range(46) if atrasos[n] == 0]
        masa = [n for n in range(46) if 1 <= atrasos[n] <= 9]
        tension = [n for n in range(46) if atrasos[n] >= 15]
        
        combinaciones = []
        if len(momento) >= 1 and len(masa) >= 4 and len(tension) >= 1:
            for m in momento[:10]:
                for mc in combinations(masa[:20], 4):
                    for t in tension[:5]:
                        combo = tuple(sorted([m] + list(mc) + [t]))
                        if len(set(combo)) == 6 and 100 <= sum(combo) <= 170:
                            combinaciones.append(combo)
                        if len(combinaciones) >= 3000:
                            break
                    if len(combinaciones) >= 3000:
                        break
                if len(combinaciones) >= 3000:
                    break
        
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
            
            # 🔧 datos_historicos usa claves enteras
            ipc_d = calcular_IPC_calibrado(combo, datos_historicos, config)
            score = ipc_d['IPC_Total'] * peso_z
            confianza = calcular_confianza(score, 
                                          max([s['score'] for s in scored] + [score]),
                                          min([s['score'] for s in scored] + [score]) if scored else score)
            scored.append({'combo': combo, 'score': score, 'confianza': confianza})
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        top_combos = scored[:top_evaluar]
        
        resultado_real = set(historial[i])
        mejores_aciertos = 0
        mejor_combo = None
        mejor_confianza = 0
        
        for item in top_combos:
            aciertos = len(set(item['combo']) & resultado_real)
            if aciertos > mejores_aciertos:
                mejores_aciertos = aciertos
                mejor_combo = item['combo']
                mejor_confianza = item['confianza']
        
        resultados_bt.append({
            'indice': i,
            'fecha': fechas[i],
            'aciertos': mejores_aciertos,
            'confianza_prediccion': mejor_confianza,
            'resultado_real': sorted(resultado_real),
            'prediccion_top': list(mejor_combo) if mejor_combo else [],
            'total_combos': len(scored)
        })
    
    progress_bar.progress(1.0)
    status_text.text("✅ Backtesting completado")
    
    return pd.DataFrame(resultados_bt)

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.title("🔬 PIV-60 v6.5 + Métricas Avanzadas")
st.markdown("**Documento:** PIP-2026-X46 | **Validación:** Backtesting Rápido + Sharpe Ratio")

tab_prediccion, tab_backtesting, tab_metricas = st.tabs([
    "🔮 Predicción Actual", 
    "⚡ Backtesting Rápido",
    "📊 Métricas Avanzadas"
])

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
        st.header("🤖 Gemini IA")
        usar_llm = st.checkbox("Activar análisis con Gemini", value=False)
        api_key_input = st.text_input("Gemini API Key", type="password", value="")
        
        st.markdown("---")
        top_n = st.slider("Top combinaciones", 5, 50, 15)
        ejecutar_pred = st.button("🚀 Ejecutar", type="primary", use_container_width=True)
    
    if ejecutar_pred:
        if archivo_historial is None or archivo_datos is None:
            st.error("❌ Sube ambos archivos")
            st.stop()
        
        try:
            with st.spinner("📥 Cargando..."):
                df_hist = pd.read_csv(archivo_historial)
                df_datos = pd.read_csv(archivo_datos)
                
            with st.spinner("⚙️ Calibrando..."):
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
                st.metric("ω Hist", f"{config['omega_hist']:.3f}")
                st.metric("ω Rec", f"{config['omega_rec']:.3f}")
                st.metric("ω Gum", f"{config['omega_gum']:.3f}")
            
            with st.spinner("🔄 Generando..."):
                datos = calibrador.datos_actuales
                historial = calibrador.historial
                fechas = calibrador.fechas
                
                # 🔧 Clasificar con claves enteras
                momento = [n for n, info in datos.items() if info['atraso'] == 0]
                masa = [n for n, info in datos.items() if 1 <= info['atraso'] <= 9]
                tension = [n for n, info in datos.items() if info['atraso'] >= 15]
                
                suma_atrasos = sum(info['atraso'] for info in datos.values())
                C_actual = suma_atrasos + config['constante_k']
                
                st.info(f"📊 Momento={len(momento)}, Masa={len(masa)}, Tensión={len(tension)}")
                st.write(f"   • Momento: {sorted(momento)}")
                st.write(f"   • Tensión: {sorted(tension)}")
                
                combinaciones = []
                for m in momento:
                    for mc in combinations(masa, 4):
                        for t in tension:
                            combo = tuple(sorted([m] + list(mc) + [t]))
                            if len(set(combo)) == 6 and 100 <= sum(combo) <= 170:
                                combinaciones.append(combo)
                
                st.write(f"✅ {len(combinaciones):,} combinaciones")
                
                resultados = []
                scores = [calcular_IPC_calibrado(c, datos, config)['IPC_Total'] for c in combinaciones]
                score_max = max(scores) if scores else 0
                score_min = min(scores) if scores else 0
                
                for combo in combinaciones:
                    # 🔧 Acceder con claves enteras
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
                    confianza = calcular_confianza(score, score_max, score_min)
                    
                    resultados.append({
                        'Rank': 0, 'Combinación': ' - '.join(map(str, combo)),
                        'N1': combo[0], 'N2': combo[1], 'N3': combo[2], 
                        'N4': combo[3], 'N5': combo[4], 'N6': combo[5],
                        'Score': score, 'IPC': ipc_d['IPC_Total'], 'Confianza': confianza,
                        'S': S, 'Zona': zona, 'Suma': sum(combo)
                    })
                
                if not resultados:
                    st.error("❌ No se generaron combinaciones.")
                    st.stop()
                
                resultados.sort(key=lambda x: x['Score'], reverse=True)
                for i, r in enumerate(resultados, 1):
                    r['Rank'] = i
                
                finalistas = resultados[:top_n]
            
            st.subheader(f"🏆 Top {top_n}")
            for i, item in enumerate(finalistas, 1):
                with st.expander(f"#{i:02d} | {item['Combinación']} | Score: {item['Score']:.4f}", expanded=(i<=5)):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Zona:** {item['Zona']}")
                        st.write(f"**S:** {item['S']:.1f}")
                        st.write(f"**Suma:** {item['Suma']}")
                        st.write(f"**Confianza:** {item['Confianza']}%")
                    with col_b:
                        st.write(f"**IPC:** {item['IPC']:.4f}")
            
            if usar_llm and api_key_input:
                st.subheader("🤖 Análisis Gemini")
                with st.spinner("🧠 Consultando..."):
                    model = get_gemini_client(api_key_input)
                    if model:
                        sumas_hist = [sum(s) for s in historial]
                        hist_stats = f"- Sorteos: {len(historial)}\n- Suma prom: {np.mean(sumas_hist):.1f}"
                        analisis = analizar_con_gemini(model, finalistas, config, hist_stats)
                        if analisis:
                            st.markdown(analisis)
            
            st.subheader(f"📋 Todas ({len(resultados):,})")
            df_resultados = pd.DataFrame(resultados)
            st.dataframe(df_resultados[['Rank', 'Combinación', 'Score', 'Confianza', 'Zona', 'Suma']], 
                        use_container_width=True, height=600)
            
            csv = df_resultados.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label="📥 Descargar CSV", data=csv, 
                              file_name=f"PIV60_Prediccion.csv", mime="text/csv")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# =============================================================================
# PESTAÑA 2: BACKTESTING RÁPIDO
# =============================================================================

with tab_backtesting:
    st.header("⚡ Backtesting Rápido (Últimos 10-20 Sorteos)")
    st.info("ℹ️ Validación acelerada en los sorteos más recientes. Resultados en ~30 segundos.")
    
    if archivo_historial is None or archivo_datos is None:
        st.warning("⚠️ Sube los archivos en 'Predicción Actual' primero")
    else:
        with st.expander("⚙️ Parámetros", expanded=True):
            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                ventana_min_bt = st.number_input("Ventana mínima", 10, 50, 20)
                top_evaluar_bt = st.number_input("Top a evaluar", 1, 50, 10)
            with col_bt2:
                max_sorteos_bt = st.slider("Máximo sorteos a evaluar", 5, 30, 15)
                ejecutar_bt = st.button("🔄 Ejecutar Backtesting", type="primary", use_container_width=True)
        
        if ejecutar_bt:
            try:
                with st.spinner("📥 Cargando..."):
                    df_hist = pd.read_csv(archivo_historial)
                    df_datos = pd.read_csv(archivo_datos)
                
                with st.spinner("⚙️ Calibrando..."):
                    calibrador = CalibradorDinamicoPIV60(df_hist, df_datos)
                    config = calibrador.ejecutar_calibracion()
                
                historial = calibrador.historial
                fechas = calibrador.fechas
                
                st.info(f"📊 Historial: {len(historial)} sorteos | Evaluando últimos {max_sorteos_bt}")
                
                df_bt = ejecutar_backtesting_rapido(historial, fechas, config, ventana_min_bt, top_evaluar_bt, max_sorteos_bt)
                
                total = len(df_bt)
                hit_3 = (df_bt['aciertos'] >= 3).sum() / total if total > 0 else 0
                hit_4 = (df_bt['aciertos'] >= 4).sum() / total if total > 0 else 0
                hit_5 = (df_bt['aciertos'] >= 5).sum() / total if total > 0 else 0
                prom = df_bt['aciertos'].mean()
                
                st.subheader("📊 Métricas")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Sorteos", total)
                with m2:
                    st.metric("Hit ≥4", f"{hit_4*100:.1f}%")
                with m3:
                    st.metric("Hit ≥5", f"{hit_5*100:.2f}%")
                with m4:
                    st.metric("Promedio", f"{prom:.2f}")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_bt['fecha'], df_bt['aciertos'], marker='o', linestyle='-', linewidth=2)
                ax.axhline(y=prom, color='r', linestyle='--', label=f"Promedio ({prom:.2f})")
                ax.axhline(y=0.78, color='gray', linestyle=':', label="Azar (0.78)")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Aciertos")
                ax.set_title("Evolución de Aciertos")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                st.subheader("📋 Resultados")
                st.dataframe(df_bt[['fecha', 'aciertos', 'confianza_prediccion', 'resultado_real', 'prediccion_top']], 
                            use_container_width=True, height=400)
                
                csv_bt = df_bt.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(label="📥 Descargar", data=csv_bt, 
                                  file_name="backtesting_rapido.csv", mime="text/csv")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# =============================================================================
# PESTAÑA 3: MÉTRICAS AVANZADAS
# =============================================================================

with tab_metricas:
    st.header("📊 Métricas Avanzadas de Rendimiento")
    
    if archivo_historial is None or archivo_datos is None:
        st.warning("⚠️ Sube los archivos primero")
    else:
        st.subheader("🎯 Configuración de Premios")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            costo = st.number_input("Costo por jugada ($)", 1, 100, 5)
            premio_3 = st.number_input("Premio 3 aciertos ($)", 0, 10000, 100)
        with col_p2:
            premio_4 = st.number_input("Premio 4 aciertos ($)", 0, 100000, 5000)
            premio_5 = st.number_input("Premio 5 aciertos ($)", 0, 1000000, 50000)
        with col_p3:
            premio_6 = st.number_input("Premio 6 aciertos ($)", 0, 10000000, 1000000)
            calcular_metricas_btn = st.button("📈 Calcular Métricas", type="primary")
        
        if calcular_metricas_btn:
            try:
                with st.spinner("📥 Procesando..."):
                    df_hist = pd.read_csv(archivo_historial)
                    df_datos = pd.read_csv(archivo_datos)
                    calibrador = CalibradorDinamicoPIV60(df_hist, df_datos)
                    config = calibrador.ejecutar_calibracion()
                    
                    df_bt = ejecutar_backtesting_rapido(calibrador.historial, calibrador.fechas, config, 20, 10, 20)
                    
                    aciertos = df_bt['aciertos'].tolist()
                    confianzas = df_bt['confianza_prediccion'].tolist()
                    
                    sharpe, retorno_prom, volatilidad = calcular_sharpe_ratio(aciertos, costo, premio_3, premio_4, premio_5, premio_6)
                    
                    total = len(df_bt)
                    hit_rates = {
                        3: ((df_bt['aciertos'] >= 3).sum() / total, premio_3),
                        4: ((df_bt['aciertos'] >= 4).sum() / total, premio_4),
                        5: ((df_bt['aciertos'] >= 5).sum() / total, premio_5),
                        6: ((df_bt['aciertos'] == 6).sum() / total, premio_6)
                    }
                    
                    ev = calcular_valor_esperado(hit_rates, {'3': hit_rates[3], '4': hit_rates[4], '5': hit_rates[5], '6': hit_rates[6]}, costo)
                    
                    correlacion = np.corrcoef(confianzas, aciertos)[0, 1] if len(confianzas) > 1 else 0
                    
                    st.subheader("📊 Resultados")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Sharpe Ratio", f"{sharpe:.3f}", 
                                 delta="Positivo" if sharpe > 0 else "Negativo")
                    with m2:
                        st.metric("Retorno Promedio", f"{retorno_prom*100:.1f}%")
                    with m3:
                        st.metric("Volatilidad", f"{volatilidad*100:.1f}%")
                    with m4:
                        st.metric("Valor Esperado", f"${ev:.2f}", 
                                 delta="Positivo" if ev > 0 else "Negativo")
                    
                    st.subheader("📝 Interpretación")
                    
                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        st.markdown("**Sharpe Ratio:**")
                        if sharpe > 1:
                            st.success("✅ Excelente (>1.0)")
                        elif sharpe > 0.5:
                            st.info("🟢 Bueno (0.5-1.0)")
                        elif sharpe > 0:
                            st.warning("🟡 Regular (0-0.5)")
                        else:
                            st.error("❌ Negativo")
                    
                    with col_i2:
                        st.markdown("**Valor Esperado:**")
                        if ev > 0:
                            st.success(f"✅ Positivo (${ev:.2f})")
                        else:
                            st.error(f"❌ Negativo (${ev:.2f})")
                    
                    st.subheader("🔗 Correlación Confianza vs. Aciertos")
                    st.write(f"**Coeficiente:** {correlacion:.3f}")
                    
                    if correlacion > 0.5:
                        st.success("✅ Fuerte correlación positiva")
                    elif correlacion > 0.2:
                        st.info("🟢 Moderada correlación")
                    else:
                        st.warning("🟡 Débil correlación")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(confianzas, aciertos, alpha=0.6, s=100, c='steelblue')
                    ax.set_xlabel("Confianza de Predicción (%)")
                    ax.set_ylabel("Aciertos Reales")
                    ax.set_title("Relación Confianza vs. Aciertos")
                    ax.grid(True, alpha=0.3)
                    
                    z = np.polyfit(confianzas, aciertos, 1)
                    p = np.poly1d(z)
                    ax.plot(confianzas, p(confianzas), "r--", alpha=0.8, label=f"Tendencia (r={correlacion:.2f})")
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    st.subheader("💡 Recomendación Final")
                    
                    if sharpe > 0.5 and ev > -costo*0.5:
                        st.success("🎯 El modelo muestra potencial. Continuar validando.")
                    elif sharpe > 0:
                        st.info("📈 Resultados mixtos. Se necesitan más datos.")
                    else:
                        st.warning("⚠️ El modelo no supera consistentemente al azar.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

st.markdown("---")
st.caption("⚠️ **Advertencia:** Este sistema es para investigación estadística. Ningún modelo puede garantizar aciertos en sorteos aleatorios. Juega responsablemente.")
